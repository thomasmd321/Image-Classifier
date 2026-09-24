''' Checks the wiki in docs/wiki: every link between pages points at an existing page and section. '''
import pathlib
import re

import pytest

WIKI = pathlib.Path(__file__).resolve().parent.parent / 'docs' / 'wiki'
LINK = re.compile(r'\]\(([A-Za-z0-9_-]+\.md)?(#[^)]*)?\)')


def anchors(page):
    ''' GitHub's heading anchors: lowercase, punctuation removed, spaces to hyphens. '''
    slugs = set()
    for line in page.read_text().splitlines():
        if line.startswith('#'):
            text = line.lstrip('#').strip().lower()
            slugs.add(re.sub(r'[^\w\- ]', '', text).replace(' ', '-'))
    return slugs


PAGES = sorted(WIKI.glob('*.md'))


def test_wiki_has_pages():
    names = {p.name for p in PAGES}
    assert {'Home.md', '_Sidebar.md'} <= names and len(names) >= 10


@pytest.mark.parametrize('page', PAGES, ids=lambda p: p.name)
def test_wiki_links_resolve(page):
    for target, anchor in LINK.findall(page.read_text()):
        if not target and not anchor:
            continue
        linked = WIKI / target if target else page
        assert linked.exists(), '{} links to missing page {}'.format(page.name, target)
        if anchor:
            assert anchor[1:] in anchors(linked), '{} links to missing section {}{}'.format(
                page.name, target, anchor)


def test_every_page_is_in_the_sidebar():
    sidebar = (WIKI / '_Sidebar.md').read_text()
    for page in PAGES:
        if page.name != '_Sidebar.md':
            assert '({})'.format(page.name) in sidebar, '{} is missing from _Sidebar.md'.format(page.name)
