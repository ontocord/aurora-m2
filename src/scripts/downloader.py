from pathlib import Path

import click
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from multiprocessing import cpu_count
from json_repair import repair_json
import json
import threading



# Create a lock object to manage concurrent access to the DataFrame
lock = threading.Lock()

# Initialize the Selenium WebDriver options
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("--start-maximized")
chrome_options.add_argument("--window-size=1920,1080")




def process_paquet_file(paraquet_file: str, out_folder: str):
    # Load the parquet file
    df = pd.read_parquet(paraquet_file)

    def get_webdriver():
        return webdriver.Chrome(options=chrome_options)

    def find_initial_view(data):
        for item in data:
            if 'initialView' in item:
                return item['initialView']
        return None

    def extract_metadata(url):
        driver = get_webdriver()
        driver.implicitly_wait(1)  # Set a shorter implicit wait
        wait = WebDriverWait(driver, 5)

        metadata = {}

        def get_element_text(selector, description):
            try:
                element = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, selector)))
                text = element.text.strip()
                print(f"Extracted {description}: {text}")
                return text
            except Exception as e:
                print(f"Could not extract {description}: {e}")
                return None

        def get_element_text_by_js(script, description):
            try:
                element = driver.execute_script(script)
                text = element.strip()
                print(f"Extracted {description}: {text}")
                return text
            except Exception as e:
                print(f"Could not extract {description}: {e}")
                return None

        try:
            print(f"Extracting metadata from {url}")
            driver.get(url)

            # Extract Title, Description, and Author
            metadata['title'] = get_element_text('h1.photo-title', 'title')
            metadata['description'] = get_element_text('h2.photo-desc p', 'description')

            # JavaScript to find the author dynamically
            author_script = """
            var authorElement = document.querySelector('a[rel="author"].owner-name');
            return authorElement ? authorElement.textContent : null;
            """
            metadata['author'] = get_element_text_by_js(author_script, 'author')

            # Extract Views, Faves, and Comments
            metadata['views'] = get_element_text('div.view-count span.view-count-label', 'views')
            metadata['faves'] = get_element_text('div.fave-count span.fave-count-label', 'faves')
            metadata['comments'] = get_element_text('div.comment-count span.comment-count-label', 'comments')

            # Extract Upload and Capture Dates
            metadata['upload_date'] = get_element_text('div.date-posted span.date-posted-label', 'upload date')
            metadata['capture_date'] = get_element_text('div.date-taken-container span.date-taken-label',
                                                        'capture date')

            # Extract License Information
            metadata['license_info'] = get_element_text('div.photo-license-info span', 'license info')

            # Extract Camera and EXIF Data
            metadata['camera'] = get_element_text('div.exif-camera-name a', 'camera info')

            # Extract Tags
            tags = []
            try:
                tag_elements = wait.until(
                    EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'ul.tags-list li a.tag-text')))
                tags = [tag.text for tag in tag_elements]
                print(f"Extracted tags: {tags}")
            except Exception as e:
                print(f"Could not extract tags: {e}")
            metadata['tags'] = ", ".join(tags)

            # Extract image URLs and resolutions from JavaScript
            try:
                script_elements = driver.find_elements(By.CSS_SELECTOR, "script")
                script_content = None
                for script_element in script_elements:
                    if 'photoModel' in script_element.get_attribute('innerHTML'):
                        script_content = script_element.get_attribute('innerHTML')
                        test = repair_json(script_content)
                        data = json.loads(test)
                        initial_view = find_initial_view(data)
                        break

                if script_content:
                    sizes = initial_view['params']['photoModel']['sizes']
                    image_urls = []

                    for size_key, size_data in sizes.items():
                        url = size_data['url']
                        width = size_data['width']
                        height = size_data['height']
                        resolution = f"{width}x{height}"
                        image_urls.append((url, resolution))

                    for idx, (url, resolution) in enumerate(image_urls):
                        clean_url = url.lstrip('\\//')
                        metadata[f'image{idx + 1}_url'] = clean_url
                        metadata[f'image{idx + 1}_resolution'] = resolution
                        print(f"Extracted image URL: {clean_url} with resolution: {resolution}")
            except Exception as e:
                print(f"Could not extract image URLs and resolutions: {e}")
        finally:
            driver.quit()

        return metadata

    def process_row(index, row, progress_bar):
        if pd.isna(df.loc[index, 'author']):
            print(f"Processing row {index}")
            progress_bar.n = index
            progress_bar.refresh()
            metadata = extract_metadata(row['url_source'])
            with lock:
                for key, value in metadata.items():
                    df.at[index, key] = value
        if index % 50 == 0:
            save_progress()
        return index

    def save_progress(dest_file: Path):
        df.to_parquet(dest_file, index=False)
        print("Progress saved")

    # Ensure the 'author' column exists
    if 'author' not in df.columns:
        df['author'] = pd.NA
    # Start from the last unprocessed row
    start_index = df[df['author'].isna()].index.min()

    with ThreadPoolExecutor(max_workers=cpu_count()) as executor:
        with tqdm(total=df.shape[0] - start_index, desc="Processing rows") as progress_bar:
            futures = {executor.submit(process_row, index, row, progress_bar): index for index, row in df.iterrows() if
                       index >= start_index}
            for future in as_completed(futures):
                index = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"Error processing row {index}: {e}")
                progress_bar.update(1)

    # Save the final DataFrame
    dst_file = Path(out_folder) / Path(paraquet_file).name
    save_progress(dst_file)
    print("Final progress saved")


@click.command()
@click.option('--paraquet-file', type=str, help='Input file.')
@click.option('--output-folder', type=str, help='Output file.')
def main(parquet_file: str, output_folder: str):
    process_paquet_file(parquet_file, output_folder)


if __name__ == '__main__':
    main()