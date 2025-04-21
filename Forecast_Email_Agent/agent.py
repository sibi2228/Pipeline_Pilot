from google.adk.agents import Agent

from google.cloud import bigquery

import pandas as pd

import matplotlib.pyplot as plt

import smtplib

from email.mime.multipart import MIMEMultipart

from email.mime.text import MIMEText

from email.mime.base import MIMEBase

from email import encoders

 

def query_bigquery(project_id, dataset_id, table_id):

    client = bigquery.Client(project=project_id)

    query = f"SELECT * FROM `{project_id}.{dataset_id}.{table_id}`"

    query_job = client.query(query)

    results = query_job.result()

    return results

 

def format_results(results):

    formatted_results = []

    for row in results:

        formatted_results.append(dict(row.items()))

    return formatted_results

 

def create_forecast_chart(project_id: str, dataset_id: str, table_id: str) -> dict:

    """Creates a line chart with forecast data from a BigQuery table."""

    try:

        results = query_bigquery(project_id, dataset_id, table_id)

        df = pd.DataFrame(format_results(results))

       

        plt.figure(figsize=(10, 6))

       

        # Plot forecast sales

        plt.plot(df['forecast_date'], df['forecasted_sales'], label='Forecast Sales')

       

        # Plot forecast range

        plt.fill_between(df['forecast_date'], df['forecast_lower_bound'], df['forecast_upper_bound'], color='gray', alpha=0.2, label='Forecast Range')

       

        plt.xlabel('Date')

        plt.ylabel('Sales')

        plt.title('Sales Forecast')

        plt.legend()

        plt.tight_layout()

        file_name = 'forecast_chart.png'

        plt.savefig(file_name)

        plt.show()

       

        return {"status": "success",

                "report": f"Forecast chart created successfully and saved as '{file_name}'."}

    except Exception as e:

        return {"status": "error",

                "error_message": str(e)}

 

def send_email(to_address: str, subject: str, body: str, attachment_path: str) -> dict:

    """Sends an email with the specified subject, body, and attachment."""

    from_address = 'romeojuli1997@gmail.com'  # Replace with your email address

    password = "khhnqtlyklreknaa"  # Replace with the generated app-specific password

   

    msg = MIMEMultipart()

    msg['From'] = from_address

    msg['To'] = to_address

    msg['Subject'] = subject

   

    msg.attach(MIMEText(body, 'plain'))

   

    try:

        attachment = open(attachment_path, "rb")

    except FileNotFoundError:

        return {"status": "error",

                "error_message": f"Attachment file '{attachment_path}' not found."}

   

    part = MIMEBase('application', 'octet-stream')

    part.set_payload(attachment.read())

    encoders.encode_base64(part)

    part.add_header('Content-Disposition', f"attachment; filename= {attachment_path}")

   

    msg.attach(part)

   

    try:

        server = smtplib.SMTP('smtp.gmail.com', 587)  # Using Gmail's SMTP server

        server.starttls()

        server.login(from_address, password)

        text = msg.as_string()

        server.sendmail(from_address, to_address, text)

        server.quit()

        return {"status": "success",

                "report": f"Email sent successfully to {to_address}."}

    except smtplib.SMTPAuthenticationError:

        return {"status": "error",

                "error_message": "Authentication failed. Please check your email address and password."}

    except smtplib.SMTPException as e:

        return {"status": "error",

                "error_message": str(e)}

root_agent = Agent(

    name="forecast_chart_email_agent",

    model="gemini-2.0-flash",

    description="Agent to create sales forecast visualizations from BigQuery data and send them via email.",

    instruction="I can create sales forecast visualizations from BigQuery data and send them via email.",

    tools=[create_forecast_chart, send_email]

)