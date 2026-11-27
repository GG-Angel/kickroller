"""Guards that hold for every route under /v1/me, including future ones."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from tests.fakes import TRACK_ID


def test_every_me_route_requires_a_user_id(
    app: FastAPI, client: TestClient
) -> None:
    del client.headers["X-User-Id"]
    # The schema lists every route; app.routes only holds included routers.
    paths = {
        path: methods
        for path, methods in app.openapi()["paths"].items()
        if path.startswith("/v1/me/")
    }
    assert paths

    for path, methods in paths.items():
        url = path.format(track_id=TRACK_ID)
        for method in methods:
            response = client.request(method, url)

            assert response.status_code == 422, (method, url)
            missing = [error["loc"] for error in response.json()["detail"]]
            assert ["header", "x-user-id"] in missing, (method, url)
