from playwright.sync_api import sync_playwright

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.set_viewport_size({"width": 1280, "height": 720})
        page.goto("http://localhost:8000/index.html")

        # Verify Proposed Searches wrap
        quick_filters = page.locator(".quick-filters")
        wrap_style = quick_filters.evaluate("el => getComputedStyle(el).flexWrap")
        print(f"Computed flex-wrap: {wrap_style}")

        # Verify Retrieve Value
        retrieve_input = page.locator("#retrievalK")
        retrieve_value = retrieve_input.input_value()
        print(f"Retrieve input value: {retrieve_value}")

        # Verify Christmas Panel Width
        christmas_banner = page.locator(".christmas-banner")
        max_width = christmas_banner.evaluate("el => getComputedStyle(el).maxWidth")
        print(f"Computed max-width: {max_width}")

        page.screenshot(path="verification_v2.png")
        browser.close()

if __name__ == "__main__":
    run()
