import asyncio
import re
from playwright.async_api import async_playwright

URL = "https://agritech.tnau.ac.in/expert_system/paddy/Schemes.html"

async def scrape_schemes():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        print("Navigating to page...")
        await page.goto(URL, timeout=60000, wait_until="networkidle")

        # Extract textual content from the page
        content = await page.inner_text("body")
        await browser.close()

        # Clean multiple blank lines
        clean_text = re.sub(r"\n{3,}", "\n\n", content).strip()

        # Save to local text file for RAG consumption
        output_file = "schemes_data.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(clean_text)

        print(f"Data successfully saved to {output_file} ({len(clean_text)} characters).")

if __name__ == "__main__":
    asyncio.run(scrape_schemes())