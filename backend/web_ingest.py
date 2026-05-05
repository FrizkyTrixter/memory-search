import os
import re
import time
import requests

from ingest import ingest_image


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STATIC_IMAGE_DIR = os.path.join(BASE_DIR, "backend", "static", "val2017")

COMMONS_API_URL = "https://commons.wikimedia.org/w/api.php"

HEADERS = {
    "User-Agent": "Mateo-MemorySearch/1.0 (local research project)"
}


def safe_filename(name):
    name = name.replace("File:", "")
    name = re.sub(r"[^a-zA-Z0-9._-]", "_", name)
    return name


def extension_from_mime(mime):
    if mime == "image/png":
        return ".png"
    if mime == "image/webp":
        return ".webp"
    if mime == "image/gif":
        return ".gif"
    return ".jpg"


def search_commons_images(query, max_images=3):
    print(f"[WEB_INGEST] Searching Commons for: {query}")

    search_params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srnamespace": 6,
        "srlimit": 8,
        "format": "json",
    }

    search_response = requests.get(
        COMMONS_API_URL,
        params=search_params,
        headers=HEADERS,
        timeout=15
    )
    search_response.raise_for_status()

    search_data = search_response.json()
    search_results = search_data.get("query", {}).get("search", [])

    print(f"[WEB_INGEST] Raw Commons results: {len(search_results)}")

    titles = [
        result.get("title")
        for result in search_results
        if result.get("title", "").startswith("File:")
    ]

    if not titles:
        return []

    info_params = {
        "action": "query",
        "titles": "|".join(titles),
        "prop": "imageinfo",
        "iiprop": "url|mime",
        "format": "json",
    }

    time.sleep(0.5)

    info_response = requests.get(
        COMMONS_API_URL,
        params=info_params,
        headers=HEADERS,
        timeout=15
    )
    info_response.raise_for_status()

    info_data = info_response.json()
    pages = info_data.get("query", {}).get("pages", {})

    images = []

    for page in pages.values():
        title = page.get("title", "")
        imageinfo = page.get("imageinfo", [])

        if not imageinfo:
            continue

        info = imageinfo[0]
        url = info.get("url", "")
        mime = info.get("mime", "")

        if not url:
            continue

        if not mime.startswith("image/"):
            continue

        images.append({
            "title": title,
            "url": url,
            "mime": mime,
        })

        print(f"[WEB_INGEST] Found image: {title} ({mime})")

        if len(images) >= max_images:
            break

    return images


def download_image(url, filename):
    os.makedirs(STATIC_IMAGE_DIR, exist_ok=True)

    save_path = os.path.join(STATIC_IMAGE_DIR, filename)

    response = requests.get(
        url,
        headers=HEADERS,
        timeout=20
    )
    response.raise_for_status()

    with open(save_path, "wb") as f:
        f.write(response.content)

    return save_path


def web_search_and_ingest(query, max_images=3):
    try:
        images = search_commons_images(query, max_images=max_images)
    except requests.exceptions.HTTPError as e:
        print("[WEB_INGEST] HTTP search failed:", e)
        return []
    except Exception as e:
        print("[WEB_INGEST] Search failed:", e)
        return []

    results = []

    for image in images:
        try:
            filename = safe_filename(image["title"])
            ext = extension_from_mime(image.get("mime", ""))

            if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".gif")):
                filename += ext

            local_path = download_image(image["url"], filename)

            ingest_image(local_path)

            results.append({
                "url": f"static/val2017/{os.path.basename(local_path)}",
                "source_url": image["url"],
                "title": image["title"],
            })

            print(f"[WEB_INGEST] Added web result: {filename}")

        except requests.exceptions.HTTPError as e:
            print("[WEB_INGEST] Failed image HTTP:", e)
        except Exception as e:
            print("[WEB_INGEST] Failed image:", e)

    print(f"[WEB_INGEST] Final web results: {len(results)}")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Search Wikimedia Commons, download images, and ingest into existing FAISS index."
    )
    parser.add_argument("query")
    parser.add_argument("--max", type=int, default=3)

    args = parser.parse_args()

    results = web_search_and_ingest(args.query, max_images=args.max)
    print(results)