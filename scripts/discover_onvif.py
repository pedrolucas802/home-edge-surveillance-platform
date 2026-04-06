from __future__ import annotations

import argparse
import logging
from urllib.parse import urlsplit, urlunsplit, quote


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Discover RTSP URLs from a camera via ONVIF")
    parser.add_argument("--host", default="192.168.0.89", help="Camera IP or hostname.")
    parser.add_argument("--port", type=int, default=80, help="ONVIF service port. Common values: 80, 8080, 8899.")
    parser.add_argument("--username", default="admin", help="Camera username.")
    parser.add_argument("--password", required=True, help="Camera password.")
    return parser


def configure_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger("discover_onvif")


def inject_credentials(uri: str, username: str, password: str) -> str:
    parts = urlsplit(uri)
    safe_username = quote(username, safe="")
    safe_password = quote(password, safe="")
    netloc = f"{safe_username}:{safe_password}@{parts.hostname or ''}"
    if parts.port is not None:
        netloc = f"{netloc}:{parts.port}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def main() -> int:
    args = build_parser().parse_args()
    logger = configure_logging()

    try:
        from onvif import ONVIFCamera
    except ImportError:
        logger.error("Missing dependency: onvif-zeep")
        logger.info("Install it with: .venv/bin/pip install onvif-zeep")
        return 1

    logger.info("Connecting to ONVIF service at %s:%s", args.host, args.port)

    try:
        camera = ONVIFCamera(args.host, args.port, args.username, args.password)
        media = camera.create_media_service()
        profiles = media.GetProfiles()
    except Exception as exc:
        logger.error("Could not query ONVIF metadata: %s", exc)
        logger.info("Try a different ONVIF port, for example 80, 8080, or 8899.")
        return 1

    if not profiles:
        logger.error("The camera returned no ONVIF media profiles.")
        return 1

    logger.info("Found %s media profile(s).", len(profiles))

    discovered = 0
    for profile in profiles:
        try:
            uri = media.GetStreamUri(
                {
                    "StreamSetup": {
                        "Stream": "RTP-Unicast",
                        "Transport": {"Protocol": "RTSP"},
                    },
                    "ProfileToken": profile.token,
                }
            )
        except Exception as exc:
            logger.warning("Failed to resolve RTSP URI for profile %s: %s", profile.token, exc)
            continue

        discovered += 1
        raw_uri = uri.Uri
        auth_uri = inject_credentials(raw_uri, args.username, args.password)
        logger.info("Profile token=%s", profile.token)
        logger.info("Raw URI: %s", raw_uri)
        logger.info("Auth URI: %s", auth_uri)

    if discovered == 0:
        logger.error("No RTSP URI could be resolved from ONVIF.")
        return 1

    logger.info("Use one of the Auth URI values above with test.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
