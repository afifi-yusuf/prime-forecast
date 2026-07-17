#!/usr/bin/env python3
"""One-time setup: AgentCore Gateway + Web Search connector (us-east-1).

Creates (or reuses) an IAM gateway role, an MCP Gateway with AWS_IAM inbound
auth, and a web-search target. Prints env vars to add to secrets.env.

Prereqs:
  - AWS credentials in the environment (IAM keys or profile) with permission
    to create IAM roles and bedrock-agentcore-control resources
  - Region must be us-east-1 (Web Search connector availability)

Usage:
  set -a && source secrets.env && set +a   # optional; needs IAM keys, not just Bedrock bearer
  python scripts/setup_agentcore_search.py
  python scripts/setup_agentcore_search.py --name prime-forecast-search
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

REGION = "us-east-1"
ROLE_NAME = "PrimeForecastAgentCoreGatewayRole"
WEB_SEARCH_ARN = f"arn:aws:bedrock-agentcore:{REGION}:aws:tool/web-search.v1"


def _account_id(sts) -> str:
    return sts.get_caller_identity()["Account"]


def _ensure_role(iam, account: str) -> str:
    role_arn = f"arn:aws:iam::{account}:role/{ROLE_NAME}"
    trust = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "bedrock-agentcore.amazonaws.com"},
            "Action": "sts:AssumeRole",
            "Condition": {
                "StringEquals": {"aws:SourceAccount": account},
                "ArnLike": {
                    "aws:SourceArn": f"arn:aws:bedrock-agentcore:{REGION}:{account}:*",
                },
            },
        }],
    }
    try:
        iam.get_role(RoleName=ROLE_NAME)
        print(f"IAM role exists: {role_arn}")
    except iam.exceptions.NoSuchEntityException:
        print(f"Creating IAM role {ROLE_NAME} ...")
        iam.create_role(
            RoleName=ROLE_NAME,
            AssumeRolePolicyDocument=json.dumps(trust),
            Description="AgentCore Gateway role for prime-forecast web search",
        )
        # Allow role to propagate
        time.sleep(8)

    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "InvokeGateway",
                "Effect": "Allow",
                "Action": "bedrock-agentcore:InvokeGateway",
                "Resource": f"arn:aws:bedrock-agentcore:{REGION}:{account}:gateway/*",
            },
            {
                "Sid": "InvokeWebSearch",
                "Effect": "Allow",
                "Action": "bedrock-agentcore:InvokeWebSearch",
                "Resource": WEB_SEARCH_ARN,
            },
        ],
    }
    iam.put_role_policy(
        RoleName=ROLE_NAME,
        PolicyName="PrimeForecastAgentCoreWebSearch",
        PolicyDocument=json.dumps(policy),
    )
    return role_arn


def _find_gateway(control, name: str) -> dict | None:
    token = None
    while True:
        kwargs = {"maxResults": 50}
        if token:
            kwargs["nextToken"] = token
        resp = control.list_gateways(**kwargs)
        for item in resp.get("items") or resp.get("gateways") or []:
            if item.get("name") == name:
                return item
        token = resp.get("nextToken")
        if not token:
            return None


def _ensure_gateway(control, name: str, role_arn: str) -> tuple[str, str]:
    existing = _find_gateway(control, name)
    if existing:
        gw_id = existing.get("gatewayId") or existing.get("gatewayIdentifier") or existing["gatewayArn"].split("/")[-1]
        print(f"Gateway exists: {gw_id}")
        desc = control.get_gateway(gatewayIdentifier=gw_id)
        url = desc.get("gatewayUrl") or desc.get("gateway", {}).get("gatewayUrl")
        if not url:
            # reconstruct
            url = f"https://gateway-{gw_id}.gateway.bedrock-agentcore.{REGION}.amazonaws.com/mcp"
        return gw_id, url

    print(f"Creating Gateway {name} ...")
    resp = control.create_gateway(
        name=name,
        roleArn=role_arn,
        protocolType="MCP",
        authorizerType="AWS_IAM",
        description="prime-forecast AgentCore web search gateway",
    )
    gw_id = resp.get("gatewayId") or resp.get("gatewayIdentifier")
    if not gw_id and "gatewayArn" in resp:
        gw_id = resp["gatewayArn"].split("/")[-1]
    # wait READY
    url = None
    for _ in range(30):
        desc = control.get_gateway(gatewayIdentifier=gw_id)
        status = desc.get("status") or desc.get("gateway", {}).get("status")
        url = desc.get("gatewayUrl") or desc.get("gateway", {}).get("gatewayUrl")
        print(f"  status={status}")
        if status in ("READY", "ACTIVE", "AVAILABLE"):
            break
        time.sleep(2)
    if not url:
        url = f"https://gateway-{gw_id}.gateway.bedrock-agentcore.{REGION}.amazonaws.com/mcp"
    return gw_id, url


def _ensure_web_search_target(control, gw_id: str) -> None:
    targets = control.list_gateway_targets(gatewayIdentifier=gw_id).get("targets") or \
        control.list_gateway_targets(gatewayIdentifier=gw_id).get("items") or []
    for t in targets:
        name = t.get("name", "")
        cfg = json.dumps(t.get("targetConfiguration") or {})
        if "web-search" in name.lower() or "web-search" in cfg:
            print(f"Web search target already present: {name}")
            return

    print("Creating web-search gateway target ...")
    control.create_gateway_target(
        gatewayIdentifier=gw_id,
        name="web-search",
        targetConfiguration={
            "mcp": {
                "connector": {
                    "source": {"connectorId": "web-search"},
                    "configurations": [{"name": "WebSearch", "parameterValues": {}}],
                }
            }
        },
        credentialProviderConfigurations=[
            {"credentialProviderType": "GATEWAY_IAM_ROLE"}
        ],
    )
    # allow provisioning
    time.sleep(5)
    print("Web search target created.")


def _smoke_search(url: str) -> None:
    """Optional smoke via MCP if mcp-proxy-for-aws is installed."""
    try:
        import asyncio
        from mcp import ClientSession
        from mcp_proxy_for_aws.client import aws_iam_streamablehttp_client
    except ImportError:
        print("Skipping live smoke (mcp / mcp-proxy-for-aws not installed).")
        return

    async def run():
        async with aws_iam_streamablehttp_client(
            endpoint=url,
            aws_service="bedrock-agentcore",
            aws_region=REGION,
            timeout=45.0,
        ) as streams:
            read, write, _ = streams
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                names = [t.name for t in (tools.tools or [])]
                print(f"Gateway tools: {names}")
                name = next((n for n in names if "search" in n.lower()), None)
                if not name:
                    raise RuntimeError("no search tool discovered")
                result = await session.call_tool(
                    name, arguments={"query": "Federal Reserve interest rates", "maxResults": 3},
                )
                print("Smoke search ok, isError=", getattr(result, "isError", None))
                for block in (result.content or [])[:1]:
                    text = getattr(block, "text", "") or str(block)
                    print("Sample:", text[:240].replace("\n", " "))

    print("Running smoke search ...")
    asyncio.run(run())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="prime-forecast-search", help="Gateway name")
    parser.add_argument("--skip-smoke", action="store_true")
    args = parser.parse_args()

    # Bedrock bearer tokens do NOT authorize AgentCore Gateway (needs IAM SigV4).
    if not (os.environ.get("AWS_ACCESS_KEY_ID") or os.environ.get("AWS_PROFILE")
            or os.environ.get("AWS_SESSION_TOKEN")):
        # still try default credential chain (SSO / instance role)
        pass

    try:
        import boto3
    except ImportError:
        print("boto3 required", file=sys.stderr)
        return 1

    session = boto3.Session(region_name=REGION)
    sts = session.client("sts")
    iam = session.client("iam")
    control = session.client("bedrock-agentcore-control", region_name=REGION)

    try:
        ident = sts.get_caller_identity()
    except Exception as e:  # noqa: BLE001
        print(
            "AWS credentials not available for IAM/AgentCore setup.\n"
            "AgentCore Gateway needs SigV4 IAM credentials (AWS_ACCESS_KEY_ID/\n"
            "AWS_SECRET_ACCESS_KEY or AWS_PROFILE). A Bedrock bearer token alone is not enough.\n"
            f"Error: {e}",
            file=sys.stderr,
        )
        return 1

    account = ident["Account"]
    print(f"Account={account} Arn={ident.get('Arn')} Region={REGION}")

    role_arn = _ensure_role(iam, account)
    gw_id, url = _ensure_gateway(control, args.name, role_arn)
    _ensure_web_search_target(control, gw_id)

    if not args.skip_smoke:
        try:
            _smoke_search(url)
        except Exception as e:  # noqa: BLE001
            print(f"Smoke search failed (gateway may still be provisioning): {e}", file=sys.stderr)

    print("\n# Add to secrets.env:")
    print("PF_SEARCH_BACKEND=agentcore")
    print(f"AGENTCORE_GATEWAY_URL={url}")
    print(f"AGENTCORE_GATEWAY_ID={gw_id}")
    print("AWS_REGION=us-east-1")
    print("# Also ensure AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY (or AWS_PROFILE)")
    print("# for an IAM principal with bedrock-agentcore:InvokeGateway on this gateway.")
    print("# The short-term AWS_BEARER_TOKEN_BEDROCK is for Bedrock Runtime only,")
    print("# not for AgentCore Gateway MCP.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
