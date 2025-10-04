from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup
from loguru import logger

from .settings import settings


class CannotGetRef(Exception): ...


def _get_html_body_by_url(client: httpx.Client, url: str) -> BeautifulSoup:
    logger.debug(f'msg="Getting {url}"')

    response = client.get(url)

    if response.status_code != 200:
        logger.error(
            f'msg="Cannot get url" {url=} status_code={response.status_code} response={response.text}'
        )
        raise CannotGetRef

    return BeautifulSoup(response.text, features="html.parser")


def parse_one_page(
    client: httpx.Client, url: str, index_url: str, already_parsed: list[str]
):
    html = _get_html_body_by_url(client=client, url=url)

    article = html.find("article", class_="bd-article")
    if not article:
        logger.warning('msg="article tag not found"')
        return

    parsed_refs = {}

    for a in article.find_all("a"):
        if href := a.get("href"):
            if (
                ".html" in href
                and "https://" not in href
                and urljoin(index_url, str(href)) not in already_parsed
            ):
                parsed_refs[href] = parse_one_page(
                    client=client,
                    url=urljoin(index_url, str(href)),
                    index_url=index_url,
                    already_parsed=list(parsed_refs.keys())
                    if not already_parsed
                    else already_parsed,
                )

    return parsed_refs


def parse_python_api(client: httpx.Client, index_url: str):
    html = _get_html_body_by_url(
        client=client, url=urljoin(index_url, "pytorch-api.html")
    )

    main_content = html.find("div", id="python-api")
    if not main_content:
        logger.error('msg="Main content not found"')
        return

    for a_tag in main_content.find_all("a"):
        if href := a_tag.get("href"):
            ref = urljoin(index_url, str(href))
            print(
                parse_one_page(
                    client=client, url=ref, index_url=index_url, already_parsed=[]
                )
            )
            break


def run_task():
    with httpx.Client() as client:
        parse_python_api(client=client, index_url=settings.torch_url)
