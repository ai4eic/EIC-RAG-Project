import streamlit as st
import bcrypt
import time
import requests
import smtplib, random, string, os
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from app_utilities import SetHeader

from streamlit_oauth import OAuth2Component
from httpx_oauth.clients.github import GitHubOAuth2



# --- Config ---
CLIENT_ID = st.secrets["github"]["client_id"]
CLIENT_SECRET = st.secrets["github"]["client_secret"]
ORG_SLUG = st.secrets["github"]["org"]              # e.g., "acme-inc"
REDIRECT_URI = st.secrets["github"]["redirect_uri"] # e.g., "https://localhost:8501"
GITHUB_SCOPES = [
    # "public_repo",      # Full control and access to public repos only

    # From here, permissions refer to both public and private repos always
    "read:org",      # Commit statuses access
    "user:email",  # Deployment statuses access
    # "repo:invite",      # Repo collaboration invite accept/decline access
    # "security_events",  # Security events through code scan access
    # "repo",             # Full control and access to repos

    # "admin:org_hook",   # R/W, ping, and delete access to owned hooks at orgs
    # "gist",             # Write access to owner's gists
    # "notifications",    # Read access to notif and full access to thread subscriptions
    # "user",             # Gives read:user, user:email, and user:follow access
    # "project",          # R/W access to projects, read:project for read-only
    # "delete_repo",      # Access to repo deletion. Must have admin access to the repo
    # "codespace",        # Full access to create and manage codespaces
    # "workflow",         # Full access to GH actions workflow files
    # "read:audit_log",   # Read access to audit logs data

    # Here permissions can be limited to write or read instead, example: "read:org"
    # "admin:repo_hook",  # R/W, ping, and delete access to repo hooks
    # "admin:org",        # Full control and access to orgs, its teams and members
    # "admin:public_key", # R/W and delete access to public keys
    # "admin:gpg_key",    # R/W and delete access to GPG keys
    # "delete:packages",  # Delete access to packages from GH Packages
]


AUTH_URL = st.secrets["github"]["AUTH_URL"]
TOKEN_URL = st.secrets["github"]["TOKEN_URL"]
API_BASE = st.secrets["github"]["API_BASE"]
SCOPES = st.secrets["github"]["SCOPES"]

def gh_get(path: str, token: str):
    r = requests.get(
        f"{API_BASE}{path}",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
        },
        timeout=20,
    )
    return r
def is_member_of_org(token: str, org: str) -> tuple[bool, dict]:
    """
    Uses the recommended endpoint:
    GET /user/memberships/orgs/{org}
    Returns (True, user_json) if active member, else (False, user_json_or_error)
    """
    # First, get the user (for display and to ensure token works)
    me = gh_get("/user", token)
    if me.status_code != 200:
        return (False, {"error": f"/user failed with {me.status_code}", "body": me.text})

    # Check membership state
    membership = gh_get(f"/user/memberships/orgs/{org}", token)
    if membership.status_code == 200:
        body = membership.json()
        return (body.get("state") == "active", me.json())
    elif membership.status_code in (302, 403):
        # 403 often indicates SSO not authorized for the org
        return (False, {"error": "Forbidden or SSO not authorized for this org.", "status": membership.status_code})
    else:
        # 404: not a member (or org slug wrong)
        return (False, {"error": f"Membership check returned {membership.status_code}", "body": membership.text})


# Assuming you have a function to check if a username exists in your database
def check_username_exists(username):
    # Implement your database check here
    pass

def gen_password(length: int):
    return ''.join(random.choices(string.ascii_uppercase + string.digits + string.ascii_lowercase, k=length))

# Assuming you have a function to send an email
def send_email(to_address: str, subject: str, message: str):
    from_address = st.secrets["admin"]["ADMIN_EMAIL"]
    password = st.secrets["admin"]["ADMIN_PASSWORD"]

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

def hash_password(password: str) -> str:
    """
    Hash a password with bcrypt + a secret pepper.
    
    Args:
        password (str): The plain-text password to hash.
    
    Returns:
        str: The bcrypt hash (including salt and cost).
    """
    combined = password.encode("utf-8")
    
    # Generate a salt (bcrypt automatically includes it in the hash)
    salt = bcrypt.gensalt()
    
    # Hash the password
    hashed = bcrypt.hashpw(combined, salt)
    
    return hashed.decode("utf-8")  # store as string

def verify_password(password: str, hashed: str) -> bool:
    """
    Verify a password against a stored bcrypt hash.
    
    Args:
        password (str): The plain-text password to verify.
        hashed (str): The stored bcrypt hash.
    
    Returns:
        bool: True if password matches, False otherwise.
    """
    combined = password.encode("utf-8")
    return bcrypt.checkpw(combined, hashed.encode("utf-8"))
def check_password(password):
    salt = st.secrets["PASSWORD_SALT"]
    # Implement your password checking logic here
    pass

def check_username(username: str):
    isValid = True
    for char in username:
        if char.isalnum() or char == "_" or char.isnumeric():
            continue
        else:
            isValid = False
            break
    return isValid


def __request_account():
    SetHeader("Request Account")
    
    with st.form("Login Form"):
        st.info("By providing your name, you agree that all the prompts and responses will be recorded and will be used to further improve RAG methods")
        name = st.text_input("What's your username? or Name if Guest")
        password = st.text_input("What's your password?", type="password")
        submitted = st.form_submit_button("Submit and start")
        if submitted:
            if (name not in st.secrets):
                st.error("User not found. Please try again or request an account")
                st.stop()
            elif not verify_password(password, st.secrets[name]["password"]):
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
            password = st.text_input("Password (Minimum 8 characters)", type = "password")
            if len(password) < 8:
                st.warning("Password must be at least 8 characters long")
                st.stop()
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
            password = hash_password(password)
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
                
                send_email(st.secrets["admin"]["ADMIN_EMAIL"], 
                        f"New account for {first_name} {last_name}",
                        Body
                )
                st.success("Your request has been sent to the admin. Shall revert back in a day.")

def request_account():
    SetHeader("Account Login/Signup")
    
    # --- Handle OAuth redirect (code + state) ---
    client = GitHubOAuth2(CLIENT_ID, CLIENT_SECRET, GITHUB_SCOPES)

    # create a button to start the OAuth2 flow
    oauth2 = OAuth2Component(client=client)

    if "github_credentials" not in st.session_state:
        result = oauth2.authorize_button(
            name="Login with Github",
            icon="https://github.githubassets.com/assets/GitHub-Mark-ea2971cee799.png",
            redirect_uri=REDIRECT_URI,
            scope=" ".join(GITHUB_SCOPES),
            key="github",
            extras_params={"prompt": "none"},
            use_container_width=True,
        )
        if result:
            with st.status("Authenticating using GitHub....", expanded = True):
                
                st.session_state["github_credentials"] = result
                st.write("Checking if you are a member of the organization....")
                
                token = result.get("token", {}).get("access_token", None)
                
                user = gh_get("/user", token)
                if user.status_code != 200:
                    st.error("Failed to retrieve Github user information. Retry later")
                    st.stop()
                elif user.status_code == 403:
                    st.error("Access forbidden. You might have exceeded the rate limit. Retry later")
                    st.json(user.json())
                else:
                    user = user.json()
                st.session_state["user"] = user

                ok, info = is_member_of_org(token, ORG_SLUG)
                
                if ok:
                    st.session_state["token"] = token
                    st.session_state["user"] = info
                    st.session_state["user_name"] = info.get("login")
                    st.session_state["authed"] = True
                    st.rerun()
                else:
                    st.error("You're not an active member of the required GitHub organization.")
                    if isinstance(info, dict) and "error" in info:
                        st.caption(f"Details: {info['error']}")
                    st.error(f"You can request to be added in {ORG_SLUG} Github Organization by submitting the form below")
                    st.rerun()
    elif not st.session_state.get("authed"):
        st.error(f"You are not an active member of {ORG_SLUG} GitHub organization.")
        st.info(f"You can request to be added in AI4EIC Github Organization by submitting the form below")
        token = st.session_state["github_credentials"].get("token", {}).get("access_token", None)
        user = st.session_state.get("user", None)
        if not user:
            user = gh_get("/user", token)
            if user.status_code != 200:
                st.error("Failed to retrieve Github user information. Retry later")
                st.stop()
            else:
                user = user.json()
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(user.get("avatar_url"), width=96)
        with col2:
            st.subheader(f"Welcome, {user.get('login')}!")
            st.caption(f"Name: {user.get('name') or '—'} • GitHub ID: {user.get('id')}")
        
        FrmCol1, FrmCol2 = st.columns([1, 1])
        with FrmCol1:
            username = st.text_input("GitHub Username", value = user.get("login"), disabled=True)
            name = st.text_input("Name", value = user.get("name", ""))
            
        with FrmCol2:
            usermail = st.text_input("Email", value = user.get("email", ""))
            institution = st.text_input("Institution", value = user.get("institution", ""))
        reason = st.text_area("Reason for requesting an account", value = st.session_state.get("Reason", ""))
        contribute = st.checkbox("Would you like to contribute to the project as well?", 
                                 value = st.session_state.get("Contribute", False), 
                                 help="If you check this box, you will be considered to be added to the contributors list"
                                 )
        if st.button("Submit"):
            st.session_state["Name"] = name
            st.session_state["Email"] = usermail
            st.session_state["Institution"] = institution
            st.session_state["Reason"] = reason
            st.session_state["Contribute"] = contribute
            metainfo = ""
            Body = f"""
                    <html>
                    <body>
                    <h2>Hello {name},</h2>
                    <p>Greetings from AI4EIC team. We have received your request to be added to the {ORG_SLUG} GitHub for RAG4EIC.</p>
                    <p>We are working on adding you to the organization and shall revert back in about a day.</p>
                    <p>Details of the request is summarized below:</p>
                    <table>
                        <tr><td>Username:</td><td>{username}</td></tr>
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
                f"New Account Request for {name} : https://github.com/{username}",
                Body
            )
            
            Body = f"""
                    <html>
                    <body>
                    <h2>Hello Admin,</h2>
                    <p>A new account has been requested for {name}.</p>
                    <p>Details of the request is summarized below:</p>
                    <table>
                    <tr><td>Username:</td><td><a href="https://github.com/{username}">{username}</a></td></tr>
                    <tr><td>Email:</td><td>{usermail}</td></tr>
                    <tr><td>Institution:</td><td>{institution}</td></tr>
                    <tr><td>Reason for requesting an account:</td><td>{reason}</td></tr>
                    <tr><td>Would you like to contribute to the project as well?</td><td>{contribute}</td></tr>
                    </table>
                    <p> To add to the organization, please go to <a href="https://github.com/orgs/{ORG_SLUG}/people">{ORG_SLUG} Members</a> and Invite the member {username}</p>
                    <p>Thank you,</p>
                    <p>AI4EIC Team</p>
                    </body>
                    </html>
            """ 
            
            send_email(st.secrets["admin"]["ADMIN_EMAIL"], 
                    f"New account for {name} : https://github.com/{username}",
                    Body
            )
            send_email(st.secrets["admin"]["AI4EIC_EMAIL"], 
                    f"New account for {name} : https://github.com/{username}",
                    Body
            )
            st.success("Your request has been sent to the admin. Shall revert back in a day.")

    else:
        st.success("You are logged in !!!!")
        st.info ("You will be redirected to the chat bot in 3 seconds")
        time.sleep(3)
        st.switch_page("pages/2_RAG-ChatBot.py")
        
    

if __name__ == "__main__":
    request_account()
