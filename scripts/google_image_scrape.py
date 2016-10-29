import time
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from urllib.request import urlretrieve
import os


QUERY = 'driver'
DOWNLOAD_DIR = '../McNulty_WIP/images/driver'

target_url_str = "https://www.google.com/search?as_st=y&tbm=isch&hl=en&as_q=%s&as_epq=&as_oq=&as_eq=&cr=&as_sitesearch=&safe=images&tbs=isz:m" % QUERY
image_xpath = "//img[@class='rg_ic rg_i']"

chromedriver = "/Users/PandaGongfu/Downloads/chromedriver"
os.environ["webdriver.chrome.driver"] = chromedriver
driver = webdriver.Chrome(chromedriver)
driver.get(target_url_str)

def hover(el):
    ActionChains(driver).move_to_element(el).perform()

def save_img_src(el, file_no, sleep_time=0.25):
    hover(el)
    time.sleep(sleep_time)

    base = el.get_attribute('src')
    if not base:
        print('no img', file_no)
        return

    file_name_full = '%s/%s.%s' % (DOWNLOAD_DIR, file_no, 'JPEG')
    try:
        urlretrieve(base, file_name_full)
        print('wrote from url %s' % file_name_full)
    except IOError as e:
        print('Bad URL?', e)
    time.sleep(sleep_time)


imgs = driver.find_elements_by_xpath(image_xpath)

for file_no, img_el in enumerate(imgs):
    save_img_src(img_el, file_no)


