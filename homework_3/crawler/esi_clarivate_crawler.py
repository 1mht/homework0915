import json
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service


# 配置参数全部从config.json读取
# 先cd data\homework_3\crawler
import os
CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'config.json'))

# 初始化浏览器
def init_driver(download_dir, chromedriver_path):
    chrome_options = Options()
    chrome_options.add_experimental_option("prefs", {
        "download.default_directory": download_dir,
        "download.prompt_for_download": False,
        "directory_upgrade": True,
        "safebrowsing.enabled": True
    })
    driver = webdriver.Chrome(service=Service(chromedriver_path), options=chrome_options)
    return driver

# 加载登录信息
def load_config(path=CONFIG_PATH):
    with open(path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config

# 登录页面并处理弹窗
def login_and_handle_popup(driver, email, password, login_url):
    driver.get(login_url)
    driver.execute_script("document.body.style.zoom='80%';")
    print("[DEBUG] 页面缩放已设置为80%")
    driver.implicitly_wait(5)
    wait = WebDriverWait(driver, 20)
    wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
    wait.until(EC.visibility_of_element_located((By.CLASS_NAME, "mat-mdc-dialog-container")))
    print("弹窗已出现")
    # PART.1 ———— 登陆表单的checkbox以及button.click一下
    try:
        driver.find_element(By.ID, "mat-mdc-checkbox-1-input").click()
        driver.find_element(By.ID, "mat-mdc-checkbox-2-input").click()
        driver.find_element(By.XPATH, "/html/body/div[2]/div[2]/div/mat-dialog-container/div/div/cross-border-data-acknowledgement/div[4]/button[2]/span[2]").click()
        time.sleep(3)
    except Exception as e:
        print("checkbox label未找到:", e)
    # PART.2 ———— 填写登录表单
    try:
        email_input = driver.find_element(By.ID, "mat-input-0")
        email_input.send_keys(email)
        password_input = driver.find_element(By.ID, "mat-input-1")
        password_input.send_keys(password)
        login_button = driver.find_element(By.ID, "signIn-btn")
        login_button.click()
    except Exception as e:
        print(f"An error occurred: {e}")

# 关闭cookies弹窗
def close_cookies(driver):
    wait = WebDriverWait(driver, 20)
    try:
        time.sleep(20)
        close_cookies_button = wait.until(EC.element_to_be_clickable((By.ID, "onetrust-close-btn-container")))
        close_cookies_button.click()
        print("已关闭cookies")
    except:
        time.sleep(20)
        close_cookies_button = wait.until(EC.element_to_be_clickable((By.ID, "onetrust-close-btn-container")))
        close_cookies_button.click()
        print("已关闭cookies")

# 搜索论文并下载csv
def download_institution_csv(driver):
    driver.execute_script("document.body.style.zoom='60%';")
    print("[DEBUG] 页面缩放为60%以提升label可见性")
    try:
        resultsList = driver.find_element(By.XPATH, "/html/body/div[1]/div[2]/div[5]/div/div/div[2]/div[1]/div/div[1]/div[2]/div/a/div")
        resultsList.click()
        time.sleep(1)
        institutions = driver.find_element(By.XPATH, "//*[@id='select2-drop']/ul/li[3]")
        institutions.click()
        time.sleep(1)
        # 下载第1个学科
        i = 1    
        # add filter     
        add_filter_button = driver.find_element(By.XPATH, "/html/body/div[1]/div[2]/div[5]/div/div/div[2]/div[1]/div/div[2]/div[2]/div[2]/a")
        add_filter_button.click()
        # 相对Xpath定位add filter二级菜单
        researchFields = WebDriverWait(driver, 20).until(EC.element_to_be_clickable((By.XPATH, "//div[@id='popup']//a[@id='researchFields']")))
        researchFields.click()
        # 选择选项1
        rf_label = driver.find_element(By.XPATH, f"/html/body/div[1]/div[2]/div[7]/div/div[2]/div/div/label[{i}]")
        rf_label.click()
        time.sleep(1)
        # 选择完关闭菜单
        back_button = driver.find_element(By.XPATH, "//*[@id='popup']/div/div[1]/a")
        back_button.click()
        time.sleep(1)
        # 点击“Download”按钮
        download_button = driver.find_element(By.XPATH, "//*[@id='action_export']")
        download_button.click()
        # 选择csv方式
        csv_button = driver.find_element(By.XPATH, "/html/body/div[1]/div[2]/div[7]/div/ul/li[2]")
        csv_button.click()
        time.sleep(3.5)
        # 下载第2到22个学科
        for i in range(2, 23):
            try:
                add_filter_button = driver.find_element(By.XPATH, "/html/body/div[1]/div[2]/div[5]/div/div/div[2]/div[1]/div/div[2]/div[2]/div[2]/a")
                add_filter_button.click()
                time.sleep(1)
                researchFields = WebDriverWait(driver, 20).until(EC.element_to_be_clickable((By.XPATH, "//div[@id='popup']//a[@id='researchFields']")))
                researchFields.click()
                time.sleep(1)
                rf_label_prev = driver.find_element(By.XPATH, f"/html/body/div[1]/div[2]/div[7]/div/div[2]/div/div/label[{i-1}]")
                rf_label_prev.click()
                time.sleep(1)
                rf_label = driver.find_element(By.XPATH, f"/html/body/div[1]/div[2]/div[7]/div/div[2]/div/div/label[{i}]")
                rf_label.click()
                time.sleep(1)
                back_button = driver.find_element(By.XPATH, "//*[@id='popup']/div/div[1]/a")
                back_button.click()
                time.sleep(1)
                download_button = driver.find_element(By.XPATH, "//*[@id='action_export']")
                download_button.click()
                time.sleep(1)
                csv_button = driver.find_element(By.XPATH, "/html/body/div[1]/div[2]/div[7]/div/ul/li[2]")
                csv_button.click()
                time.sleep(3.5)
            except Exception as e:
                print(f"An error occurred in iteration {i}: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

# 主流程
def main():
    config = load_config()
    email = config['email']
    password = config['password']
    download_dir = config['download_dir']
    chromedriver_path = config['chromedriver_path']
    login_url = config['login_url']
    driver = init_driver(download_dir, chromedriver_path)
    print(f"[DEBUG] 当前设置的Chrome下载目录: {download_dir}")
    try:
        actual_dir = driver.execute_script("return navigator.userAgent;")
        print(f"[DEBUG] Chrome UserAgent: {actual_dir}")
    except Exception as e:
        print(f"[DEBUG] 获取UserAgent失败: {e}")
    try:
        login_and_handle_popup(driver, email, password, login_url)
        close_cookies(driver)
        download_institution_csv(driver)
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        driver.quit()

if __name__ == "__main__":
    main()
