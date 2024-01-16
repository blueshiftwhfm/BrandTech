from langchain.document_loaders.sitemap import SitemapLoader
from bs4 import BeautifulSoup


def remove_nav_and_header_elements(content: BeautifulSoup) -> str:
    # Find all 'nav' and 'header' elements in the BeautifulSoup object
    nav_elements = content.find_all("nav")
    header_elements = content.find_all("header")
    # Remove each 'nav' and 'header' element from the BeautifulSoup object

    for element in nav_elements + header_elements:
        element.decompose()
    return str(content.get_text())


caminho_ifood = 'https://www.news.ifood.com.br/sitemap.xml'
caminho_btg = 'https://www.btgpactual.com/sitemap-pages.xml'

sitemap_loader_ifood = SitemapLoader(web_path=caminho_ifood, parsing_function=remove_nav_and_header_elements, filter_urls=[
                                     'https://news.ifood.com.br/institucional/'])
sitemap_loader_btg = SitemapLoader(web_path=caminho_btg, parsing_function=remove_nav_and_header_elements, filter_urls=[
                                   'https://www.btgpactual.com/nosso-dna', 'https://www.btgpactual.com/nosso-dna/valores', 'https://www.btgpactual.com/nosso-dna/proposito'])

docs_ifood = sitemap_loader_ifood.load()
docs_btg = sitemap_loader_btg.load()


# print(docs_ifood[0].page_content)


# Usar a url para visitar o site usando o GPT
