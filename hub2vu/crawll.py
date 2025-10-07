# !pip install selenium beautifulsoup4 lxml pandas

import time
import random
from urllib.parse import urljoin

import pandas as pd
from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


BASE_URL = "https://finance.naver.com/news/news_list.naver?mode=LSS3D&section_id=101&section_id2=258&section_id3=401"

def make_driver(headless=True):
    chrome_opts = Options()
    if headless:
        chrome_opts.add_argument("--headless=new")
    chrome_opts.add_argument("--no-sandbox")
    chrome_opts.add_argument("--disable-dev-shm-usage")
    chrome_opts.add_argument("--window-size=1280,2000")
    chrome_opts.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
    driver = webdriver.Chrome(options=chrome_opts)
    driver.set_page_load_timeout(30)
    return driver


def get_top_article_links(driver, max_cnt=10):
    """상위 리스트 페이지에서 맨 위부터 기사 링크 n개 추출"""
    driver.get(BASE_URL)

    # 페이지 로드/렌더 대기
    WebDriverWait(driver, 15).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "li.newsList dd.articleSubject a"))
    )
    time.sleep(0.5)  # 살짝 더 대기

    # BeautifulSoup으로 파싱
    soup = BeautifulSoup(driver.page_source, "lxml")

    # 안내 주신 구조: li.newsList.top 하단 dd.articleSubject a
    # (혹시 top이 없는 항목도 함께 들어있을 수 있어, 폭넓게 선택 후 상위부터 10개)
    candidates = soup.select("li.newsList.top dd.articleSubject a")
    if not candidates:
        # fallback: top 클래스가 없거나 구조가 다를 때
        candidates = soup.select("li.newsList dd.articleSubject a")

    links = []
    seen = set()

    for a in candidates:
        href = a.get("href")
        title = a.get_text(strip=True)
        if not href:
            continue
        url = urljoin(BASE_URL, href)
        # 중복 제거
        if url in seen:
            continue
        seen.add(url)
        links.append({"title": title, "url": url})
        if len(links) >= max_cnt:
            break

    return links


def fetch_article_body(driver, url):
    """기사 페이지에서 div#newsct_article.newsct_article._article_body 본문 텍스트 추출"""
    driver.get(url)
    # 본문 컨테이너 대기
    try:
        WebDriverWait(driver, 12).until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, "div#newsct_article.newsct_article._article_body, div#newsct_article")
            )
        )
    except Exception:
        # 일부 페이지는 다른 템플릿일 수 있으니 한 번 더 잠깐 대기
        time.sleep(1)

    soup = BeautifulSoup(driver.page_source, "lxml")

    # 가장 우선: 지정하신 정확한 선택자
    body = soup.select_one("div#newsct_article.newsct_article._article_body")
    if body is None:
        # fallback: id만 매칭되는 경우
        body = soup.select_one("div#newsct_article")

    if body:
        # 불필요한 스크립트/광고 요소 제거(안전하게 p 태그만 모으는 것도 방법)
        # 여기서는 간단히 텍스트만 추출
        text = body.get_text("\n", strip=True)
        return text
    return ""


def main():
    driver = make_driver(headless=True)
    try:
        articles = get_top_article_links(driver, max_cnt=10)
        results = []

        for i, art in enumerate(articles, 1):
            print(f"[{i}] Fetching: {art['title']}  ({art['url']})")
            body = fetch_article_body(driver, art["url"])
            results.append(
                {
                    "rank": i,
                    "title_from_list": art["title"],
                    "url": art["url"],
                    "body": body,
                }
            )
            # 매너있는 딜레이 (0.6~1.2s 랜덤)
            time.sleep(random.uniform(0.6, 1.2))

        df = pd.DataFrame(results)
        # 결과 확인
        pd.set_option("display.max_colwidth", 120)
        print(df[["rank", "title_from_list", "url"]].head(10))
        # 필요 시 CSV로 저장
        df.to_csv("naver_finance_top10.csv", index=False, encoding="utf-8-sig")
        return df

    finally:
        driver.quit()


if __name__ == "__main__":
    df = main()
