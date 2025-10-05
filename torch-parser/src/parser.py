import os
import re
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup
from html_to_markdown import convert_to_markdown
from loguru import logger

from .settings import settings


def _get_html_body_by_url(client: httpx.Client, url: str) -> BeautifulSoup:
    logger.debug(f'msg="Getting {url}"')

    response = client.get(url)

    if response.status_code != 200:
        logger.error(
            f'msg="Cannot get url" {url=} status_code={response.status_code} response={response.text}'
        )
        return BeautifulSoup("")

    return BeautifulSoup(response.text, features="lxml")


def parse_content(client: httpx.Client, url: str) -> str:
    html = _get_html_body_by_url(client, url)

    article = html.find("article", class_="bd-article", id=False)
    if not article:
        logger.warning('msg="article tag not found"')
        return ""

    md = convert_to_markdown(article.prettify())

    # fix backslashs
    md = md.replace("\\", "")

    # remove permanentlinks to headers
    md = re.sub(r"\[#\].+", "", md)

    # fix [[]] for source
    md = md.replace("[[source]]", "[source]")

    return md


def parse_one_page(client: httpx.Client, url: str, index_url: str, path_to_save: str):
    html = _get_html_body_by_url(client=client, url=url)

    article = html.find("article", class_="bd-article")
    if not article:
        logger.warning('msg="article tag not found"')
        return

    parsed_refs = {}

    for a in article.find_all("a"):
        if href := a.get("href"):
            if ".html" in href and "https://" not in href:
                current_ref = urljoin(index_url, str(href))

                with open(
                    path_to_save + f"{str(href).split('/')[-1].split('.html')[0]}.md",
                    "w",
                ) as f:
                    f.write(parse_content(client, current_ref))

    return parsed_refs


def run_task():
    with httpx.Client() as client:
        html = _get_html_body_by_url(
            client=client, url=urljoin(settings.torch_url, "pytorch-api.html")
        )

        article = html.find("article", class_="bd-article")
        if not article:
            logger.warning('msg="article tag not found in python api"')
            return

        div = article.find("div", class_="toctree-wrapper compound")
        if not div:
            logger.warning('msg="Not found main div in python api article"')
            return

        for a_tag in div.find_all("a"):
            if href := a_tag.get("href"):
                ref = urljoin(settings.torch_url, str(href))
                path_to_save = f"{settings.path_to_save}/{str(href).split('.html')[0]}/"

                if not os.path.exists(path_to_save):
                    os.mkdir(path_to_save)

                parse_one_page(
                    client=client,
                    url=ref,
                    index_url=settings.torch_url,
                    path_to_save=path_to_save,
                )
