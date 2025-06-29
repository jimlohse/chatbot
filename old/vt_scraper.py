import requests
from bs4 import BeautifulSoup
import time
import json
from datetime import datetime
from urllib.parse import urljoin, urlparse
import re
from typing import List, Dict

class VTRailroadScraper:
    def __init__(self):
        self.base_url = "https://www.virginiatruckee.com"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

        # Key pages to scrape based on the search results
        self.key_pages = [
            "/",  # Homepage
            "/about-us",
            "/contact-us/"
            "/filmmakers-photographers/",
            "/whos-been-working-on-the-railroad/",
            "/vt-history-by-stephen-drew/",
            "/schedule-fares/",  # Schedule and fares
            "/maps-directions/comstock-train-route-round-trip-to-gold-hill/",  # Comstock route
            "/maps-directions/carson-train-route/",  # Carson route
            "/maps-directions/",  # Maps and directions
            "/maps-directions/historic-original-1870-depot-map-photo-gallery/",  # Historic depot
            "/vt-equipment-roster-locomotives/",  # Equipment roster
            "/theme-trains-events/",  # Theme trains (if exists)
            "/theme-trains-events/halloween-steam-train/",
            "/theme-trains-events/holiday-train-o-lights/",
            "/theme-trains-events/pumpkin-patch-trains/",
            "/theme-trains-events/vt-candy-cane-express/",  # Candy Cane Express
            "/special-event-theme-trains/"
        ]

    def clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        if not text:
            return ""

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters that might cause issues
        text = re.sub(r'[^\w\s.,!?;:()\-$%]', '', text)
        return text.strip()

    def extract_page_content(self, url: str) -> Dict:
        """Extract content from a single page"""
        try:
            print(f"Scraping: {url}")
            response = self.session.get(url, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()

            # Extract title
            title = soup.find('title')
            title_text = title.get_text() if title else ""

            # Extract main content - try different content containers
            content_selectors = [
                'main',
                '.content',
                '.main-content',
                'article',
                '.post-content',
                '#content',
                'body'
            ]

            content_text = ""
            for selector in content_selectors:
                content_container = soup.select_one(selector)
                if content_container:
                    content_text = content_container.get_text()
                    break

            if not content_text:
                # Fallback: get all text from body
                content_text = soup.get_text()

            # Clean the content
            cleaned_content = self.clean_text(content_text)

            # Extract meta description
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            description = meta_desc.get('content', '') if meta_desc else ''

            return {
                'url': url,
                'title': self.clean_text(title_text),
                'description': self.clean_text(description),
                'content': cleaned_content,
                'scraped_at': datetime.now().isoformat(),
                'word_count': len(cleaned_content.split())
            }

        except requests.RequestException as e:
            print(f"Error scraping {url}: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error scraping {url}: {e}")
            return None

    def chunk_content(self, content: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split content into overlapping chunks for better RAG retrieval"""
        if not content:
            return []

        words = content.split()
        chunks = []

        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = ' '.join(chunk_words)
            chunks.append(chunk_text)

            # Break if we've reached the end
            if i + chunk_size >= len(words):
                break

        return chunks

    def scrape_all_pages(self) -> List[Dict]:
        """Scrape all key pages and return structured data"""
        all_content = []

        for page_path in self.key_pages:
            full_url = urljoin(self.base_url, page_path)
            page_data = self.extract_page_content(full_url)

            if page_data and page_data['content']:
                # Chunk the content
                chunks = self.chunk_content(page_data['content'])

                # Create a record for each chunk
                for i, chunk in enumerate(chunks):
                    chunk_data = {
                        'id': f"{page_path}_{i}",
                        'url': page_data['url'],
                        'title': page_data['title'],
                        'description': page_data['description'],
                        'content': chunk,
                        'chunk_index': i,
                        'total_chunks': len(chunks),
                        'scraped_at': page_data['scraped_at'],
                        'page_path': page_path
                    }
                    all_content.append(chunk_data)

                print(f"✓ Scraped {len(chunks)} chunks from {page_path}")
            else:
                print(f"✗ Failed to scrape {page_path}")

            # Be polite to the server
            time.sleep(1)

        return all_content

    def save_to_json(self, content: List[Dict], filename: str = 'vt_railroad_content.json'):
        """Save scraped content to JSON file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(content, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(content)} content chunks to {filename}")

    def discover_additional_pages(self) -> List[str]:
        """Discover additional pages by following links from the homepage"""
        try:
            response = self.session.get(self.base_url)
            soup = BeautifulSoup(response.content, 'html.parser')

            # Find all internal links
            links = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                if href.startswith('/') or self.base_url in href:
                    full_url = urljoin(self.base_url, href)
                    # Filter out unwanted links
                    if not any(x in full_url.lower() for x in ['#', 'mailto:', 'tel:', 'javascript:', '.pdf', '.jpg', '.png']):
                        links.append(full_url)

            # Remove duplicates and sort
            unique_links = list(set(links))
            print(f"Discovered {len(unique_links)} additional pages")
            return unique_links

        except Exception as e:
            print(f"Error discovering pages: {e}")
            return []

# Example usage
if __name__ == "__main__":
    scraper = VTRailroadScraper()

    print("Starting Virginia & Truckee Railroad website scraping...")
    print("=" * 50)

    # Scrape all key pages
    content = scraper.scrape_all_pages()

    # Save to JSON
    scraper.save_to_json(content)

    print("\n" + "=" * 50)
    print(f"Scraping complete! Total chunks: {len(content)}")

    # Optional: Discover additional pages for future scraping
    print("\nDiscovering additional pages...")
    additional_pages = scraper.discover_additional_pages()
    if additional_pages:
        print(f"Found {len(additional_pages)} additional pages that could be scraped in the future")