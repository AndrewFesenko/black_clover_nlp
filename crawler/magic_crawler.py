import scrapy
from bs4 import BeautifulSoup

class BlogSpider(scrapy.Spider):
    name = 'blackcloverspider'
    start_urls = ['https://blackclover.fandom.com/wiki/Category:Spells']

    def parse(self, response):
        # Extract all the links on the page
        for title in response.css('.category-page__members a::attr(href)').extract():
            spell_url = response.urljoin(title)
            
            # Determine if this is a category or a spell page
            if "Category:" in title:
                # If it's a category, follow it recursively
                yield scrapy.Request(spell_url, callback=self.parse)
            else:
                # If it's a spell page, follow it to extract data
                yield scrapy.Request(spell_url, callback=self.parse_spell)

    def parse_spell(self, response):
        # Extract spell name
        spell_name = response.css("span.mw-page-title-main::text").extract_first().strip()

        # Get the main content of the page
        div_selector = response.css("div.mw-parser-output")[0]
        div_html = div_selector.extract()

        # Use BeautifulSoup to process the HTML
        soup = BeautifulSoup(div_html, 'html.parser')

        # Extract spell type
        spell_type = ""
        aside = soup.find('aside')
        if aside:
            for cell in aside.find_all('div', {'class': 'pi-data'}):
                if cell.find('h3'):
                    cell_name = cell.find('h3').text.strip()
                    if cell_name == "Parent Magic":
                        spell_type = cell.find('div').text.strip()

        # Remove aside to clean up the soup object
        if aside:
            aside.decompose()

        # Now, find the Description header using the id 'Description'
        spell_description = ""
        description_header = soup.find('span', id='Description')

        if description_header:
            # Look for the next <p> tag that contains the description text
            next_element = description_header.find_next('p')
            if next_element:
                spell_description = next_element.get_text(strip=True)

        # Return extracted data
        return dict(
            spell_name=spell_name,
            spell_type=spell_type,
            spell_description=spell_description
        )
