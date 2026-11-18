import hashlib
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pywikibot
from pywikibot.site import BaseSite
from tqdm import tqdm

from clean_wiki import clean_wiki_pages

WIKI_URL = "https://wiki.gtnewhorizons.com/wiki/Main_Page"
GIT_URL = "https://github.com/GTNewHorizons/GT-New-Horizons-Modpack/tree/master"
GIT_QUESTS_PATH = "/config/betterquesting/DefaultQuests/Quests"


def get_site(url: str = WIKI_URL) -> BaseSite:
    return pywikibot.Site(url=url)


def fetch_page(site: BaseSite, title: str) -> pywikibot.Page | None:
    page = pywikibot.Page(site, title)
    if not page.exists():
        return None
    if page.isRedirectPage():
        page = page.getRedirectTarget()
    return page


def iter_pages(
    site: BaseSite,
    namespace: int = 0,
    limit: int | None = None,
) -> Iterator[pywikibot.Page]:
    yield from site.allpages(
        namespace=namespace,
        filterredir=False,
        content=True,
        total=limit,
    )


def download_wiki_page(path: Path, page: pywikibot.Page) -> bool:
    if path.exists():
        local_data = path.read_text(encoding="utf-8")
        downloaded_hash = hashlib.md5(page.text.encode()).hexdigest()
        local_hash = hashlib.md5(local_data.encode()).hexdigest()
        if local_hash == downloaded_hash:
            return False
    else:
        path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(page.text, encoding="utf-8")

    return True


def download_wiki_pages(root: Path, limit: int | None = 5) -> int:
    site = get_site()
    count_downloaded = 0
    pages = iter_pages(site, namespace=0, limit=limit)
    for item in tqdm(pages, total=limit, desc="Downloading wiki pages", unit="page"):
        title = item.title()

        if title.lower().startswith((".", "version")):
            continue

        filepath = root / Path(f"{title}.txt")

        if download_wiki_page(filepath, item):
            tqdm.write(f"- {title} ({len(item.text)} chars)")
            count_downloaded += 1

    return count_downloaded


def _parse_git_url(url: str) -> tuple[str, str]:
    if "/tree/" in url:
        base, branch = url.split("/tree/", maxsplit=1)
        return f"{base.rstrip('/')}.git", branch.strip("/")
    return url, "master"


def download_git_quests(root: Path) -> int:
    repo_url, branch = _parse_git_url(GIT_URL)
    sparse_path = GIT_QUESTS_PATH.lstrip("/")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_repo = Path(tmp_dir)
        subprocess.run(
            [
                "git",
                "clone",
                "--depth=1",
                "--filter=blob:none",
                "--no-checkout",
                "--branch",
                branch,
                repo_url,
                str(tmp_repo),
            ],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(tmp_repo), "sparse-checkout", "init", "--cone"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(tmp_repo), "sparse-checkout", "set", sparse_path],
            check=True,
        )
        subprocess.run(["git", "-C", str(tmp_repo), "checkout", branch], check=True)

        quests_src = tmp_repo / sparse_path
        if root.exists():
            shutil.rmtree(root)
        shutil.copytree(quests_src, root)

    return sum(1 for p in root.rglob("*") if p.is_file())


def main() -> None:
    WIKI_PATH = Path("data/wiki/")
    WIKI_CLEAN_PATH = Path("data/wiki_clean/")
    QUESTS_PATH = Path("data/quests/")

    # download_wiki_pages(WIKI_PATH, None)
    # download_git_quests(QUESTS_PATH)
    clean_wiki_pages(WIKI_PATH, WIKI_CLEAN_PATH)


if __name__ == "__main__":
    main()
