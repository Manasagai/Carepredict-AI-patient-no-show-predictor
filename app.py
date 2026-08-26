"""CarePredict AI: a demonstration dashboard for appointment attendance."""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="CarePredict AI", page_icon="🏥", layout="wide")

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_FILE = PROJECT_ROOT / "models" / "random_forest.pkl"
FEATURE_FILE = PROJECT_ROOT / "models" / "feature_columns.pkl"
DATA_FILE = PROJECT_ROOT / "data" / "processed_appointments.csv"


def add_styles():
	st.markdown("""
	<style>
	:root { --ink: #16324f; --teal: #167d83; --gold: #efb366; }
	.stApp { background: linear-gradient(135deg, #f5fbfa 0%, #ffffff 52%, #eef5fb 100%); }
	[data-testid="stSidebar"] { background: #16324f; }
	[data-testid="stSidebar"] * { color: #f5fbfa; }
	[data-testid="stMetric"] { background: white; border: 1px solid #dcebe9; border-radius: 10px; padding: 16px; box-shadow: 0 3px 14px #16324f10; }
	.hero { background: linear-gradient(110deg, #16324f, #167d83); color: white; border-radius: 14px; padding: 30px 34px; margin-bottom: 24px; }
	.hero h1 { color: white; margin: 0 0 8px; font-size: 2.25rem; }
	.hero p { color: #d9f0ec; margin: 0; font-size: 1.05rem; }
	.section-title { color: #16324f; border-left: 5px solid #efb366; padding-left: 12px; margin: 28px 0 16px; }
	.feature-card { background: white; border: 1px solid #dcebe9; border-radius: 10px; padding: 18px; min-height: 130px; }
	.feature-card h3 { color: #167d83; margin-top: 0; }
	.auth-wrap { max-width: 560px; margin: 8vh auto 0; }
	</style>
	""", unsafe_allow_html=True)


@st.cache_resource
def load_model_artifacts():
	return joblib.load(MODEL_FILE), joblib.load(FEATURE_FILE)


@st.cache_data
def load_data():
	return pd.read_csv(DATA_FILE)


def age_group(age):
	if age <= 17:
		return "0-17"
	if age <= 30:
		return "18-30"
	if age <= 45:
		return "31-45"
	if age <= 60:
		return "46-60"
	return "61+"


def action_for_probability(probability):
	if probability < 0.30:
		return "Standard Reminder"
	if probability < 0.60:
		return "Early Reminder"
	return "Priority Follow-up"


def action_explanation(action):
	return {
		"Standard Reminder": "Use the normal reminder workflow.",
		"Early Reminder": "Contact the patient earlier and allow extra time to confirm.",
		"Priority Follow-up": "Prioritize direct follow-up and appointment confirmation.",
	}[action]


def create_model_input(values, feature_columns):
	"""Create an encoded row in precisely the saved training-column order."""
	row = pd.DataFrame(0, index=[0], columns=feature_columns, dtype=int)
	row["Gender"] = values["Gender"]
	row["Age"] = values["Age"]
	row["Scholarship"] = int(values["Scholarship"])
	row["Hipertension"] = int(values["Hypertension"])
	row["Diabetes"] = int(values["Diabetes"])
	row["Alcoholism"] = int(values["Alcoholism"])
	row["Handcap"] = int(values["Handicap"])
	row["SMS_received"] = int(values["SMS received"])
	row["Waiting_Days"] = values["Waiting_Days"]
	row["Appointment_Month"] = values["Appointment_Month"]
	row["Appointment_Hour"] = values["Appointment_Hour"]
	for category in (f"Neighbourhood_{values['Neighbourhood']}", f"Appointment_Day_{values['Appointment_Day']}", f"Age_Group_{values['Age_Group']}"):
		if category in row.columns:
			row[category] = 1
	return row


def no_show_probability(model, model_input):
	no_show_index = list(model.classes_).index(1)
	return float(model.predict_proba(model_input)[0, no_show_index])


def show_authentication():
	st.markdown('<div class="auth-wrap">', unsafe_allow_html=True)
	st.markdown('<div class="hero"><h1>CarePredict AI</h1><p>AI-Powered Patient Appointment Intelligence</p></div>', unsafe_allow_html=True)
	st.write("Predict smarter. Remind earlier. Reduce missed appointments.")
	auth_mode = st.radio("Account access", ["Login", "Sign Up"], horizontal=True)
	if auth_mode == "Login":
		with st.form("login_form"):
			email = st.text_input("Email", placeholder="name@hospital.org")
			password = st.text_input("Password", type="password")
			login = st.form_submit_button("Login", type="primary", use_container_width=True)
		if login:
			if email.strip() and password:
				st.session_state.authenticated = True
				st.session_state.user_email = email.strip()
				st.rerun()
		st.caption("Demo access: any non-empty email and password will work.")
	else:
		with st.form("signup_form"):
			full_name = st.text_input("Full Name")
			email = st.text_input("Email", placeholder="name@hospital.org")
			password = st.text_input("Password", type="password")
			confirm_password = st.text_input("Confirm Password", type="password")
			role = st.selectbox("Role", ["Clinician", "Care coordinator", "Hospital administrator", "Student researcher"])
			create_account = st.form_submit_button("Create Account", type="primary", use_container_width=True)
		if create_account:
			if not full_name.strip() or not email.strip() or not password:
				st.warning("Complete all required fields to create the demo account.")
			elif password != confirm_password:
				st.warning("Passwords do not match.")
			else:
				st.session_state.authenticated = True
				st.session_state.user_email = email.strip()
				st.session_state.user_role = role
				st.rerun()
	st.markdown('</div>', unsafe_allow_html=True)


def load_required_assets():
	try:
		data = load_data()
		model, feature_columns = load_model_artifacts()
		return data, model, feature_columns
	except FileNotFoundError as error:
		st.error(f"A required project file is missing: {error.filename}")
		st.stop()
	except (OSError, ValueError, EOFError, pd.errors.ParserError) as error:
		st.error(f"Project data could not be loaded: {error}")
		st.stop()


def dashboard_metrics(data):
	total = len(data)
	no_shows = int(data["No-show"].sum())
	return total, no_shows, (no_shows / total * 100 if total else 0), int(data["SMS_received"].sum())


def show_home(data, model, feature_columns):
	total, no_shows, no_show_rate, sms_count = dashboard_metrics(data)
	st.markdown('<div class="hero"><h1>Welcome to CarePredict AI</h1><p>AI-powered appointment intelligence for smarter hospital scheduling.</p></div>', unsafe_allow_html=True)
	metrics = st.columns(4)
	metrics[0].metric("Total Appointments", f"{total:,}")
	metrics[1].metric("Total No-Shows", f"{no_shows:,}")
	metrics[2].metric("No-Show Rate", f"{no_show_rate:.1f}%")
	metrics[3].metric("SMS Reminders", f"{sms_count:,}")
	st.markdown('<h2 class="section-title">Today\'s Appointment Intelligence</h2>', unsafe_allow_html=True)
	queue = reminder_records(data, model, feature_columns)
	counts = queue["Recommended Action"].value_counts()
	intelligence = st.columns(4)
	intelligence[0].metric("Total appointments", f"{len(queue):,}")
	intelligence[1].metric("Priority follow-ups", f"{counts.get('Priority Follow-up', 0):,}")
	intelligence[2].metric("Early reminders", f"{counts.get('Early Reminder', 0):,}")
	intelligence[3].metric("Standard reminders", f"{counts.get('Standard Reminder', 0):,}")
	st.subheader("Show-up vs No-show")
	st.bar_chart(pd.DataFrame({"Appointments": [total - no_shows, no_shows]}, index=["Show-up", "No-show"]), color="#167d83")
	st.markdown('<h2 class="section-title">How CarePredict AI Helps</h2>', unsafe_allow_html=True)
	features = [("Predict", "Estimate no-show probability from appointment and patient details."), ("Prioritize", "Surface appointments that may benefit from direct follow-up."), ("Remind", "Match each appointment to a practical reminder action."), ("Understand", "Turn attendance patterns into useful operational insight.")]
	columns = st.columns(4)
	for column, (title, description) in zip(columns, features):
		with column:
			st.markdown(f'<div class="feature-card"><h3>{title}</h3><p>{description}</p></div>', unsafe_allow_html=True)
	st.write("")
	st.button("Start an AI prediction", type="primary", on_click=lambda: setattr(st.session_state, "page", "🔮 AI Prediction"))


def prediction_form(model, feature_columns):
	neighbourhoods = sorted(column.removeprefix("Neighbourhood_") for column in feature_columns if column.startswith("Neighbourhood_"))
	with st.form("ai_prediction_form"):
		st.markdown("#### Patient Information")
		patient_columns = st.columns(3)
		with patient_columns[0]: age = st.number_input("Age", 0, 120, 35, 1)
		with patient_columns[1]: gender_label = st.selectbox("Gender", ["Female", "Male"])
		with patient_columns[2]: neighbourhood = st.selectbox("Neighbourhood", neighbourhoods)
		st.markdown("#### Appointment Details")
		appointment_columns = st.columns(3)
		with appointment_columns[0]: scheduled_date = st.date_input("Scheduled Date")
		with appointment_columns[1]: appointment_date = st.date_input("Appointment Date")
		with appointment_columns[2]: scheduled_hour = st.number_input("Scheduled hour", 0, 23, 9, 1, help="The trained model uses the hour of the scheduled time.")
		st.caption("Waiting Days and Appointment Day are calculated automatically from the dates.")
		st.markdown("#### Health & Support")
		health_columns = st.columns(6)
		with health_columns[0]: scholarship = st.checkbox("Scholarship")
		with health_columns[1]: hypertension = st.checkbox("Hypertension")
		with health_columns[2]: diabetes = st.checkbox("Diabetes")
		with health_columns[3]: alcoholism = st.checkbox("Alcoholism")
		with health_columns[4]: handicap = st.checkbox("Handicap")
		with health_columns[5]: sms_received = st.checkbox("SMS Received")
		submitted = st.form_submit_button("Predict No-Show Probability", type="primary", use_container_width=True)
	if not submitted:
		return
	try:
		scheduled = pd.Timestamp(scheduled_date).replace(hour=scheduled_hour)
		appointment = pd.Timestamp(appointment_date)
		waiting_days = (appointment.normalize() - scheduled.normalize()).days
		if waiting_days < 0:
			st.error("Appointment Date must be on or after Scheduled Date.")
			return
		values = {"Age": age, "Gender": 0 if gender_label == "Female" else 1, "Neighbourhood": neighbourhood, "Scholarship": scholarship, "Hypertension": hypertension, "Diabetes": diabetes, "Alcoholism": alcoholism, "Handicap": handicap, "SMS received": sms_received, "Waiting_Days": waiting_days, "Appointment_Day": appointment.day_name(), "Appointment_Month": appointment.month, "Appointment_Hour": scheduled.hour, "Age_Group": age_group(age)}
		probability = no_show_probability(model, create_model_input(values, feature_columns))
	except (ValueError, TypeError, KeyError) as error:
		st.error(f"The appointment details could not be processed: {error}")
		return
	action = action_for_probability(probability)
	st.divider()
	result_columns = st.columns(4)
	result_columns[0].metric("AI Prediction", f"{probability * 100:.1f}%")
	result_columns[1].metric("No-show probability", action)
	result_columns[2].metric("Waiting days", waiting_days)
	result_columns[3].metric("Appointment day", appointment.day_name())
	st.success(f"Recommended action: {action}")
	st.write(action_explanation(action))


def show_prediction(model, feature_columns):
	st.title("🔮 AI Prediction")
	st.caption("Use the trained Random Forest model to estimate appointment attendance and select a reminder action.")
	prediction_form(model, feature_columns)


def reminder_records(data, model, feature_columns):
	features = data[feature_columns].copy()
	probabilities = model.predict_proba(features)[:, list(model.classes_).index(1)]
	records = pd.DataFrame({"Age": data["Age"], "Gender": data["Gender"].map({0: "Female", 1: "Male"}), "Waiting Days": data["Waiting_Days"], "SMS Received": data["SMS_received"].map({0: "No", 1: "Yes"}), "No-show Probability": probabilities})
	records["Recommended Action"] = records["No-show Probability"].map(action_for_probability)
	age_columns = [column for column in data if column.startswith("Age_Group_")]
	records["Age Group"] = data[age_columns].idxmax(axis=1).str.removeprefix("Age_Group_")
	if "AppointmentID" in data:
		records.insert(0, "Appointment ID", data["AppointmentID"])
	else:
		records.insert(0, "Appointment ID", [f"Record {index + 1}" for index in range(len(data))])
	return records.sort_values("No-show Probability", ascending=False).reset_index(drop=True)


def show_reminder_center(data, model, feature_columns):
	st.title("📋 Smart Reminder Center")
	st.caption("Prioritize the available appointment dataset using the existing Random Forest model.")
	queue = reminder_records(data, model, feature_columns)
	counts = queue["Recommended Action"].value_counts()
	summary = st.columns(3)
	summary[0].metric("Priority Follow-ups", f"{counts.get('Priority Follow-up', 0):,}")
	summary[1].metric("Early Reminders", f"{counts.get('Early Reminder', 0):,}")
	summary[2].metric("Standard Reminders", f"{counts.get('Standard Reminder', 0):,}")
	filters = st.columns(3)
	with filters[0]: action_filter = st.multiselect("Recommended Action", sorted(queue["Recommended Action"].unique()))
	with filters[1]: sms_filter = st.selectbox("SMS Received", ["All", "Yes", "No"])
	with filters[2]: age_filter = st.multiselect("Age group", sorted(queue["Age Group"].unique()))
	filtered = queue.copy()
	if action_filter: filtered = filtered[filtered["Recommended Action"].isin(action_filter)]
	if sms_filter != "All": filtered = filtered[filtered["SMS Received"] == sms_filter]
	if age_filter: filtered = filtered[filtered["Age Group"].isin(age_filter)]
	display_columns = ["Appointment ID", "Age", "Gender", "Waiting Days", "SMS Received", "No-show Probability", "Recommended Action"]
	st.dataframe(filtered[display_columns].style.format({"No-show Probability": "{:.1%}"}), use_container_width=True, hide_index=True)
	st.download_button("Download reminder priority list", filtered[display_columns].to_csv(index=False).encode("utf-8"), "carepredict_reminder_priority.csv", "text/csv")


def percentage_by(data, group_column, label="No-show %"):
	return data.groupby(group_column, dropna=False)["No-show"].mean().mul(100).sort_values(ascending=False).rename(label).to_frame()


def show_analytics(data):
	st.title("📊 Appointment Analytics")
	st.caption("Explore attendance patterns in the processed appointment dataset.")
	data = data.copy()
	data["Gender Label"] = data["Gender"].map({0: "Female", 1: "Male"})
	data["SMS Label"] = data["SMS_received"].map({0: "No", 1: "Yes"})
	data["Waiting Days Group"] = pd.cut(data["Waiting_Days"], [-1, 0, 7, 14, 30, float("inf")], labels=["Same day", "1-7 days", "8-14 days", "15-30 days", "31+ days"])
	for prefix, output in [("Age_Group_", "Age Group"), ("Neighbourhood_", "Neighbourhood"), ("Appointment_Day_", "Appointment Day")]:
		columns = [column for column in data if column.startswith(prefix)]
		if columns: data[output] = data[columns].idxmax(axis=1).str.removeprefix(prefix)
	charts = [("No-show percentage by Age Group", "Age Group"), ("No-show percentage by Gender", "Gender Label"), ("No-show percentage by SMS Received", "SMS Label"), ("No-show percentage by Waiting Days Group", "Waiting Days Group"), ("No-show percentage by Appointment Day", "Appointment Day"), ("No-show percentage by Neighbourhood", "Neighbourhood")]
	for start in range(0, len(charts), 2):
		columns = st.columns(2)
		for column, (title, group) in zip(columns, charts[start:start + 2]):
			with column:
				st.subheader(title)
				st.bar_chart(percentage_by(data, group), color="#167d83")
				st.caption("Percentage of appointments marked as no-shows within each group.")
	st.markdown('<h2 class="section-title">Key Findings</h2>', unsafe_allow_html=True)
	findings = []
	for group, description in [("Age Group", "age group"), ("Gender Label", "gender"), ("SMS Label", "SMS group"), ("Waiting Days Group", "waiting-time group"), ("Appointment Day", "appointment day"), ("Neighbourhood", "neighbourhood")]:
		result = percentage_by(data, group)
		if not result.empty: findings.append(f"The highest no-show rate is in the {result.index[0]} {description} ({result.iloc[0, 0]:.1f}%).")
	for finding in findings[:4]: st.write(f"• {finding}")


def show_insights(model, feature_columns):
	st.title("🧠 Model Insights")
	st.caption("A transparent view of the loaded Random Forest model and its learned feature influence.")
	model_columns = st.columns(4)
	model_columns[0].metric("Model name", "Random Forest")
	model_columns[1].metric("Number of trees", f"{len(model.estimators_):,}")
	model_columns[2].metric("ROC-AUC", "Unavailable")
	model_columns[3].metric("Precision / Recall / F1", "Unavailable")
	st.caption("Evaluation metrics were not saved as model artifacts, so no values are inferred here.")
	importance = pd.DataFrame({"Feature": feature_columns, "Importance": model.feature_importances_}).sort_values("Importance", ascending=False)
	st.subheader("Top 10 influential features")
	st.bar_chart(importance.head(10).set_index("Feature"), color="#167d83")
	st.dataframe(importance.head(10).style.format({"Importance": "{:.4f}"}), use_container_width=True, hide_index=True)
	st.markdown('<h2 class="section-title">How the AI works</h2>', unsafe_allow_html=True)
	st.markdown("**Appointment Data**  →  **Data Cleaning**  →  **Feature Engineering**  →  **Random Forest**  →  **No-Show Probability**  →  **Reminder Recommendation**")


def main():
	add_styles()
	if "authenticated" not in st.session_state: st.session_state.authenticated = False
	if "page" not in st.session_state: st.session_state.page = "🏠 Home"
	if not st.session_state.authenticated:
		show_authentication()
		return
	data, model, feature_columns = load_required_assets()
	st.sidebar.markdown("# CarePredict AI")
	st.sidebar.caption("AI-Powered Patient Appointment Intelligence")
	st.sidebar.markdown(f"Signed in as **{st.session_state.get('user_email', 'demo user')}**")
	pages = ["🏠 Home", "🔮 AI Prediction", "📋 Smart Reminder Center", "📊 Appointment Analytics", "🧠 Model Insights"]
	st.session_state.page = st.sidebar.radio("Main navigation", pages, index=pages.index(st.session_state.page))
	st.sidebar.divider()
	if st.sidebar.button("Logout", use_container_width=True):
		st.session_state.authenticated = False
		st.rerun()
	if st.session_state.page == "🏠 Home": show_home(data, model, feature_columns)
	elif st.session_state.page == "🔮 AI Prediction": show_prediction(model, feature_columns)
	elif st.session_state.page == "📋 Smart Reminder Center": show_reminder_center(data, model, feature_columns)
	elif st.session_state.page == "📊 Appointment Analytics": show_analytics(data)
	else: show_insights(model, feature_columns)


if __name__ == "__main__":
	main()
