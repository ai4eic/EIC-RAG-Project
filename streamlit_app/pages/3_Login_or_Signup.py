import streamlit as st
import time
import smtplib, random, string, os
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from app_utilities import SetHeader

# Assuming you have a function to check if a username exists in your database
def check_username_exists(username):
    # Implement your database check here
    pass

def gen_password(length: int):
    return ''.join(random.choices(string.ascii_uppercase + string.digits + string.ascii_lowercase, k=length))

# Assuming you have a function to send an email
def send_email(to_address: str, subject: str, message: str):
    from_address = st.secrets["ADMIN_EMAIL"]
    password = st.secrets["ADMIN_PASSWORD"]

    msg = MIMEMultipart()
    msg['From'] = from_address
    msg['To'] = to_address
    msg['Subject'] = subject

    body = message
    msg.attach(MIMEText(body, 'html'))
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(from_address, password)
    text = msg.as_string()
    server.sendmail(from_address, to_address, text)
    server.quit()
    
    

def check_username(username: str):
    isValid = True
    for char in username:
        if char.isalnum() or char == "_" or char.isnumeric():
            continue
        else:
            isValid = False
            break
    return isValid


def request_account():
    SetHeader("Request Account")
    
    with st.form("Login Form"):
        st.info("By providing your name, you agree that all the prompts and responses will be recorded and will be used to further improve RAG methods")
        name = st.text_input("What's your username?")
        password = st.text_input("What's your password?", type="password")
        submitted = st.form_submit_button("Submit and start")
        if submitted:
            userInfo = st.secrets.get#get_user_info(os.environ["USER_DB"], name)
            if (name not in st.secrets):
                st.error("User not found. Please try again or request an account")
                st.stop()
            elif (password != st.secrets[name]["password"]):
                st.error("Incorrect password. Please try again")
                st.stop()
            else:
                for key in st.session_state:
                    del st.session_state[key]
                st.session_state["user_name"] = name
                st.session_state["first_name"] =  st.secrets[name]["first_name"]
                st.session_state["last_name"] =  st.secrets[name]["last_name"]
                st.success("Welcome {} {}!!! Will redirect to the chat bot in 3 seconds......".format(st.session_state.get("first_name", ""), st.session_state.get("last_name", "")))
                st.session_state["user_mode"] = int(st.secrets[name]["mode"])
                time.sleep(2)
                st.switch_page("pages/2_RAG-ChatBot.py")
        
    with st.expander("Request an Account"):
        if st.session_state.get("user_name"):
            st.info("You are already logged in !!!!")
            st.stop()
        FrmCol1, FrmCol2 = st.columns([1, 1])
        with FrmCol1:
            first_name = st.text_input("First Name", value = st.session_state.get("FirstName", ""))
            last_name = st.text_input("Last Name", value = st.session_state.get("LastName", ""))
            institution = st.text_input("Institution", value = st.session_state.get("Institution", ""))
        with FrmCol2:
            username = st.text_input("Username Combination of alphabets, numbers and underscores")
            password = st.text_input("Password (Randomly generate if empty)", type = "password")
            usermail = st.text_input("Email", value = st.session_state.get("Email", ""))
        reason = st.text_area("Reason for requesting an account", value = st.session_state.get("Reason", ""))
        contribute = st.checkbox("Would you like to contribute to the project as well?", 
                                 value = st.session_state.get("Contribute", False), 
                                 help="If you check this box, you will be added to the contributors list"
                                 )
        if st.button("Submit"):
            st.session_state["FirstName"] = first_name
            st.session_state["LastName"] = last_name
            st.session_state["Username"] = username
            st.session_state["Email"] = usermail
            st.session_state["Institution"] = institution
            st.session_state["Reason"] = reason
            st.session_state["Contribute"] = contribute
            password = password.replace(" ", "").replace("\t", "")
            if (password == ""):
                password = gen_password(random.randint(10, 20))
            if not check_username(username):
                st.error("Username must be a combination of alphabets, numbers and underscores.")
            elif username in st.secrets:
                st.error("Username already exists. Please choose a different one.")
            else:
                metainfo = ""
                Body = f"""
                        <html>
                        <body>
                        <h2>Hello {first_name} {last_name},</h2>
                        <p>Greetings from AI4EIC team. We have received your request for an account.</p>
                        <p>We are working on setting up the account for you and shall revert back in about a day.</p>
                        <p>Details of the request is summarized below:</p>
                        <table>
                            <tr><td>Username:</td><td>{username}</td></tr>
                            <tr><td>Password:</td><td>{password}</td></tr>
                            <tr><td>Email:</td><td>{usermail}</td></tr>
                            <tr><td>Institution:</td><td>{institution}</td></tr>
                            <tr><td>Reason for requesting an account:</td><td>{reason}</td></tr>
                            <tr><td>Would you like to contribute to the project as well?</td><td>{contribute}</td></tr>
                        </table>
                        
                        <p>If you have any questions or clarifications, Please reply back to this email.</p>
                        <p>Thank you,</p>
                        <p>AI4EIC Team</p>
                        </body>
                        </html>
                        """
                send_email(usermail,
                    f"New Account Request for {last_name} {first_name}",
                    Body
                )
                UserInfo = {"FirstName": first_name, "LastName": last_name, "Username": username, "Email": usermail, "Institution": institution, "Password": password, "Reason": reason} 
                Body = f"""
                        <html>
                        <body>
                        <h2>Hello Admin,</h2>
                        <p>A new account has been requested for {first_name} {last_name}.</p>
                        <p>Details of the request is summarized below:</p>
                        <table>
                        <tr><td>Username:</td><td>{username}</td></tr>
                        <tr><td>Password:</td><td>{password}</td></tr>
                        <tr><td>Email:</td><td>{usermail}</td></tr>
                        <tr><td>Institution:</td><td>{institution}</td></tr>
                        <tr><td>Reason for requesting an account:</td><td>{reason}</td></tr>
                        <tr><td>Would you like to contribute to the project as well?</td><td>{contribute}</td></tr>
                        </table>
                        <p> To include in secrets.toml include if approved is </p>
                        <p> [{username}] <br> first_name="{first_name}" <br> last_name="{last_name}" <br> user_name="{username}" <br> password="{password}" <br> email="{usermail}" <br> institution="{institution}" <br> mode="FILL_THIS" </p>
                        <p>Thank you,</p>
                        <p>AI4EIC Team</p>
                        </body>
                        </html>
                """ 
                
                send_email(st.secrets["ADMIN_EMAIL"], 
                        f"New account for {first_name} {last_name}",
                        Body
                )
                st.success("Your request has been sent to the admin. Shall revert back in a day.")
if __name__ == "__main__":
    request_account()