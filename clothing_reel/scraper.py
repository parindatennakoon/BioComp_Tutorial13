import json
import re
import anthropic
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup


MAX_PRODUCTS = 12


def scrape_page(url: str) -> tuple[str, list[dict]]:
    """Scrape a URL and return (page_html, inline_images_data)."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_extra_http_headers({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        })
        page.goto(url, wait_until="networkidle", timeout=30000)
        page.wait_for_timeout(2000)
        # Scroll to trigger lazy loading
        page.evaluate("window.scrollTo(0, document.body.scrollHeight / 2)")
        page.wait_for_timeout(1000)
        html = page.content()
        browser.close()
    return html


def extract_products_with_claude(html: str, base_url: str) -> list[dict]:
    """Use Claude to extract clothing product data from scraped HTML."""
    soup = BeautifulSoup(html, "html.parser")

    # Remove scripts, styles, and nav noise
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    # Build a condensed text + image-src snapshot
    img_tags = soup.find_all("img", src=True)
    img_srcs = []
    for img in img_tags:
        src = img.get("src", "") or img.get("data-src", "")
        if src and not src.startswith("data:") and len(src) > 4:
            # Resolve relative URLs
            if src.startswith("//"):
                src = "https:" + src
            elif src.startswith("/"):
                from urllib.parse import urlparse
                parsed = urlparse(base_url)
                src = f"{parsed.scheme}://{parsed.netloc}{src}"
            img_srcs.append(src)

    # Grab visible text (first 8000 chars to stay within token limits)
    text_content = soup.get_text(separator="\n", strip=True)[:8000]

    prompt = f"""You are analyzing an online clothing shop's HTML page.
Extract ALL clothing products visible on this page.

For each product, return a JSON object with these exact keys:
- "name": product name (string)
- "price": price with currency symbol (string, e.g. "$49.99" or "€29.00")
- "image_url": the most relevant product image URL from the list below (string)

Rules:
- Only include actual clothing/fashion products (clothes, shoes, bags, accessories)
- Skip navigation links, banners, and non-product images
- If price is not visible, use "Price unavailable"
- Pick the best image URL from the provided list for each product
- Return a JSON array of up to {MAX_PRODUCTS} products

Page text (truncated):
{text_content}

Available image URLs found on the page:
{json.dumps(img_srcs[:80], indent=2)}

Return ONLY a valid JSON array, no markdown, no explanation."""

    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip()

    # Strip markdown code fences if present
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)

    try:
        products = json.loads(raw)
        if isinstance(products, list):
            return products[:MAX_PRODUCTS]
    except json.JSONDecodeError:
        # Try to extract JSON array from response
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())[:MAX_PRODUCTS]
            except json.JSONDecodeError:
                pass

    return []


def scrape_products(url: str) -> list[dict]:
    """Main entry point: scrape a shop URL and return list of products."""
    html = scrape_page(url)
    products = extract_products_with_claude(html, url)
    # Filter out products with no image
    products = [p for p in products if p.get("image_url")]
    return products
