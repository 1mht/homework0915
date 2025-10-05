from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
import time
import json


import os
# 使用相对路径
download_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'downloads'))
chrome_options = Options()
chrome_options.add_experimental_option("prefs", {
    "download.default_directory": download_dir,
    "download.prompt_for_download": False,
    "directory_upgrade": True,
    "safebrowsing.enabled": True
})

chromedriver_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'chromedriver-win64', 'chromedriver.exe'))
service = Service(chromedriver_path)
driver = webdriver.Chrome(service=service, options=chrome_options)

# DEBUG: 输出当前下载目录配置，确认是否生效
print(f"[DEBUG] 当前设置的Chrome下载目录: {download_dir}")
try:
    actual_dir = driver.execute_script("return navigator.userAgent;")
    print(f"[DEBUG] Chrome UserAgent: {actual_dir}")
except Exception as e:
    print(f"[DEBUG] 获取UserAgent失败: {e}")

login_url = "https://access.clarivate.com/login?app=esi"

# def debug_screenshot(driver, step):
#     driver.save_screenshot(f"debug_{step}.png")
#     print(f"已保存截图: debug_{step}.png")

try:
    driver.get(login_url)
        # 页面加载后立即设置缩放比例，提升所有元素可见性
    driver.execute_script("document.body.style.zoom='80%';")
    print("[DEBUG] 页面缩放已设置为80%")
    # 隐式、显示等待
    driver.implicitly_wait(5)
    wait = WebDriverWait(driver, 20)
    wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
    # 等待弹窗容器出现
    wait.until(EC.visibility_of_element_located((By.CLASS_NAME, "mat-mdc-dialog-container")))
    print("弹窗已出现")


    # PART.1 ———— 点击checkbox的label以及button
    # 点击checkbox的label
    try:
        driver.find_element(By.ID, "mat-mdc-checkbox-1-input").click()
        driver.find_element(By.ID, "mat-mdc-checkbox-2-input").click()
        driver.find_element(By.XPATH, "/html/body/div[2]/div[2]/div/mat-dialog-container/div/div/cross-border-data-acknowledgement/div[4]/button[2]/span[2]").click()
        time.sleep(3)

    except Exception as e:
        print("checkbox label未找到:", e)


    # PART.2 ———— 填写登录表单
    try:
        # 加载用户名以及密码
        config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'config.json'))
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        email = config['email']
        password = config['password']

        # 填写邮箱
        email_input = driver.find_element(By.ID, "mat-input-0")
        email_input.send_keys(email)

        # 填写密码
        password_input = driver.find_element(By.ID, "mat-input-1")
        password_input.send_keys(password)

        # 提交表单
        login_button = driver.find_element(By.ID, "signIn-btn")
        login_button.click()
    except Exception as e:
        print(f"An error occurred: {e}")
 
    
    ## 登录后的操作
    # PART.3 ———— 关闭cookies
    try:
        # 关闭接受cookies的按钮
        time.sleep(20)  # 等待页面加载
        close_cookies_button = wait.until(EC.element_to_be_clickable((By.ID, "onetrust-close-btn-container")))
        close_cookies_button.click()
        print("已关闭cookies")
    except:
        time.sleep(20)
        close_cookies_button = wait.until(EC.element_to_be_clickable((By.ID, "onetrust-close-btn-container")))
        close_cookies_button.click()
        print("已关闭cookies")

    # PART.4 ———— 搜索论文
    # 修改页面缩放比例，提升所有label可见性
    driver.execute_script("document.body.style.zoom='60%';")
    print("[DEBUG] 页面缩放为60%以提升label可见性")
    try:
        # results list
        resultsList = driver.find_element(By.XPATH, "/html/body/div[1]/div[2]/div[5]/div/div/div[2]/div[1]/div/div[1]/div[2]/div/a/div")
        resultsList.click()
        time.sleep(1)
        # 选择“institutions”
        institutions = driver.find_element(By.XPATH, "//*[@id='select2-drop']/ul/li[3]")
        institutions.click()
        time.sleep(1)

        # when i = 1, select the first research field and download csv
        i = 1
        # add filter 
        add_filter_button = driver.find_element(By.XPATH, "/html/body/div[1]/div[2]/div[5]/div/div/div[2]/div[1]/div/div[2]/div[2]/div[2]/a")
        add_filter_button.click()

        ''' 定位add filter二级菜单 '''
        # 相对Xpath定位方式
        researchFields = wait.until(EC.element_to_be_clickable((By.XPATH, "//div[@id='popup']//a[@id='researchFields']")))
        researchFields.click()

        # # 选择选项1  
        rf_label = driver.find_element(By.XPATH, f"/html/body/div[1]/div[2]/div[7]/div/div[2]/div/div/label[{i}]")
        rf_label.click()
        time.sleep(1)
        # # 选择完关闭菜单
        back_button = driver.find_element(By.XPATH, "//*[@id='popup']/div/div[1]/a")
        back_button.click()
        time.sleep(1)

        # 点击“Download”按钮
        download_button = driver.find_element(By.XPATH, "//*[@id='action_export']")
        download_button.click()
        # # 选择csv方式
        csv_button = driver.find_element(By.XPATH, "/html/body/div[1]/div[2]/div[7]/div/ul/li[2]")
        csv_button.click()
        time.sleep(3.5)

        # 依次选择第2到第22个领域 并下载csv文档
        for i in range(2, 23):
            try:
                '''定位不到add filter三级菜单的label'''
                # add filter 
                add_filter_button = driver.find_element(By.XPATH, "/html/body/div[1]/div[2]/div[5]/div/div/div[2]/div[1]/div/div[2]/div[2]/div[2]/a")
                add_filter_button.click()
                time.sleep(1)

                ''' 定位add filter二级菜单 '''
                # 相对Xpath定位方式
                researchFields = wait.until(EC.element_to_be_clickable((By.XPATH, "//div[@id='popup']//a[@id='researchFields']")))
                researchFields.click()
                time.sleep(1)

                # 反选选项i-1
                rf_label_prev = driver.find_element(By.XPATH, f"/html/body/div[1]/div[2]/div[7]/div/div[2]/div/div/label[{i-1}]")
                rf_label_prev.click()
                time.sleep(1)

                # 选择选项i
                rf_label = driver.find_element(By.XPATH, f"/html/body/div[1]/div[2]/div[7]/div/div[2]/div/div/label[{i}]")
                rf_label.click()
                time.sleep(1)

                # 等待表格加载
                time.sleep(1)

                # # 选择完关闭菜单
                back_button = driver.find_element(By.XPATH, "//*[@id='popup']/div/div[1]/a")
                back_button.click()
                time.sleep(1)

                # 点击“Download”按钮
                download_button = driver.find_element(By.XPATH, "//*[@id='action_export']")
                download_button.click()
                time.sleep(1)

                # # 选择csv方式
                csv_button = driver.find_element(By.XPATH, "/html/body/div[1]/div[2]/div[7]/div/ul/li[2]")
                csv_button.click()
                time.sleep(3.5)

            except Exception as e:
                print(f"An error occurred in iteration {i}: {e}")

        # time.sleep(30)
    except Exception as e:
        print(f"An error occurred: {e}")

except Exception as e:
    print(f"An error occurred: {e}")



