#!/usr/bin/env python3
"""
Test script for the Virginia & Truckee Railroad scraper
Run this to test the scraper and see what content it extracts
"""

from vt_scraper import VTRailroadScraper
import json

def test_single_page():
    """Test scraping a single page first"""
    scraper = VTRailroadScraper()
    
    print("Testing single page scrape...")
    test_url = "https://www.virginiatruckee.com/"
    
    page_data = scraper.extract_page_content(test_url)
    if page_data:
        print(f"SUCCESS: Successfully scraped homepage")
        print(f"Title: {page_data['title']}")
        print(f"Word count: {page_data['word_count']}")
        print(f"Content preview: {page_data['content'][:200]}...")
        
        # Test chunking
        chunks = scraper.chunk_content(page_data['content'])
        print(f"Split into {len(chunks)} chunks")
        return True
    else:
        print("FAILED: Failed to scrape homepage")
        return False

def main():
    """Main test function"""
    print("Virginia & Truckee Railroad Scraper Test")
    print("=" * 50)
    
    # Test single page first
    if not test_single_page():
        print("Single page test failed. Check your internet connection and try again.")
        return
    
    print("\n" + "=" * 50)
    print("Single page test successful! Running full scrape...")
    
    # Run full scrape
    scraper = VTRailroadScraper()
    content = scraper.scrape_all_pages()
    
    if content:
        # Save results
        scraper.save_to_json(content, 'test_vt_content.json')
        
        # Show summary
        print(f"\nSCRAPING SUMMARY:")
        print(f"Total content chunks: {len(content)}")
        print(f"Pages scraped: {len(set(item['page_path'] for item in content))}")
        
        # Show sample content
        print(f"\nSAMPLE CONTENT:")
        for i, chunk in enumerate(content[:3]):
            print(f"\nChunk {i+1}:")
            print(f"Page: {chunk['page_path']}")
            print(f"Title: {chunk['title']}")
            print(f"Content: {chunk['content'][:150]}...")
        
        print(f"\nSUCCESS: Test complete! Content saved to 'test_vt_content.json'")
        print(f"You can now proceed to step 3 (embedding pipeline)")
        
    else:
        print("ERROR: No content was scraped. Check the scraper configuration.")

if __name__ == "__main__":
    main()
