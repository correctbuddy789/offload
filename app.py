import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
import asyncio
import re

st.title("📊 Salary Content Generator | Built at Grapevine")

# ----- Define Synonyms for Robust Matching -----
YOE_SYNONYMS = ["yoe", "years", "experience"]
TITLE_SYNONYMS = ["job_title", "title", "role", "position", "designation"]
SALARY_SYNONYMS = ["total", "salary", "compensation", "ctc", "annual"]

def find_best_column_match(columns_list, synonyms):
    for col in columns_list:
        col_lower = col.lower().strip()
        for syn in synonyms:
            if syn in col_lower:
                return col
    return None

def detect_and_rename_columns(df):
    remaining_columns = list(df.columns)
    # Detect Year of Experience
    yoe_col = find_best_column_match(remaining_columns, YOE_SYNONYMS)
    if yoe_col:
        remaining_columns.remove(yoe_col)
    else:
        raise ValueError("Missing required column: Year of Experience")
    # Detect Job Title
    title_col = find_best_column_match(remaining_columns, TITLE_SYNONYMS)
    if title_col:
        remaining_columns.remove(title_col)
    else:
        raise ValueError("Missing required column: Job Title")
    # Detect Salary
    salary_col = find_best_column_match(remaining_columns, SALARY_SYNONYMS)
    if not salary_col:
        raise ValueError("Missing required column: Salary")
    remaining_columns.remove(salary_col)
    # Optionally detect Location column using synonyms
    location_synonyms = ["location", "city", "area"]
    location_col = find_best_column_match(remaining_columns, location_synonyms)
    new_names = {yoe_col: "YOE", title_col: "Job_Title", salary_col: "Salary"}
    if location_col:
        new_names[location_col] = "Location"
    return df.rename(columns=new_names)

def extract_yoe_from_string(job_title_string):
    if pd.isna(job_title_string):
        return None
    match = re.search(r'\((\d+\.?\d*)\s*yoe\)', job_title_string, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None

def clean_data(df, manual_yoe_column=None):
    if df.empty:
        raise ValueError("Uploaded CSV file is empty.")
    df = detect_and_rename_columns(df)
    if manual_yoe_column and manual_yoe_column in df.columns:
        df['YOE'] = df[manual_yoe_column]
    elif df['YOE'].isna().all():
        extracted = df['Job_Title'].apply(extract_yoe_from_string)
        if extracted.notna().any():
            df['YOE'] = extracted
    def convert_salary(val):
        try:
            if pd.isna(val):
                return np.nan
            if isinstance(val, (int, float)):
                return float(val)
            if isinstance(val, str):
                cleaned_val = ''.join(c for c in val if c.isdigit() or c == '.')
                return float(cleaned_val)
        except (ValueError, TypeError):
            return np.nan
    df['Salary'] = df['Salary'].apply(convert_salary)
    df.dropna(subset=['Salary', 'Job_Title'], inplace=True)
    df['YOE'] = pd.to_numeric(df['YOE'], errors='coerce')
    df.dropna(subset=['YOE'], inplace=True)
    # Round YOE to whole number for display
    df['YOE'] = df['YOE'].round(0)
    return df

def clean_company_name(company):
    # Remove unwanted substring starting with " - query_result" (case-insensitive)
    return re.sub(r'\s*-\s*query_result.*$', '', company, flags=re.IGNORECASE)

def format_salary_entry(row):
    # For standard (viral) posts
    job_title = row['Job_Title']
    yoe = row['YOE']
    salary_in_lakhs = row['Salary'] / 100000
    if salary_in_lakhs >= 100:
        salary_str = f"{salary_in_lakhs / 100:.2f} Crores"
    else:
        salary_str = f"{salary_in_lakhs:.1f} Lakhs"
    return f"{job_title} with {int(yoe)} YOE earns {salary_str}"

def format_salary_entry_custom(row):
    # For custom query posts (includes company info)
    job_title = row['Job_Title']
    job_title = re.sub(r'\s*-\s*query_result.*$', '', job_title, flags=re.IGNORECASE)
    yoe = row['YOE']
    salary_in_lakhs = row['Salary'] / 100000
    company = row["Company"] if "Company" in row else ""
    company = clean_company_name(company)
    if salary_in_lakhs >= 100:
        salary_str = f"{salary_in_lakhs / 100:.2f} Crores"
    else:
        salary_str = f"{int(salary_in_lakhs)} LPA"
    return f"{job_title} at {company} with {int(yoe)} YOE earns {salary_str}"

def select_viral_entries(df):
    job_counts = df['Job_Title'].value_counts()
    top_titles = job_counts.head(5).index.tolist()
    selected_rows = []
    for title in top_titles:
        subset = df[df['Job_Title'] == title]
        rep_row = subset.sort_values('Salary', ascending=False).iloc[0]
        selected_rows.append(rep_row)
    return pd.DataFrame(selected_rows)

def parse_query(query_text):
    """
    Parses a natural language query.
    For example, from:
      'Software Engineer with experience of 5 years and need 2 salaries'
    it extracts:
      - roles: ['Software Engineer']
      - experience: 5
      - location: None (if not specified)
      - salary_count: 2 (if specified)
    """
    roles = []
    experience = None
    location = None
    salary_count = None

    # Extract experience (e.g., "5 years")
    exp_match = re.search(r'(\d+)\s*(?:years|yrs)', query_text, re.IGNORECASE)
    if exp_match:
        experience = float(exp_match.group(1))

    # Extract salary count (e.g., "need 2 salaries" or "show me 3 salaries")
    count_match = re.search(r'\b(?:need|show me|give me)\s+(\d+)\s*salaries', query_text, re.IGNORECASE)
    if count_match:
        salary_count = int(count_match.group(1))

    # Use "salaries" pattern if present; otherwise split on "with"
    if "salaries" in query_text.lower():
        role_match = re.search(r'^(.*?)\s+salaries', query_text, re.IGNORECASE)
        if role_match:
            role_str = role_match.group(1).strip()
            role_str = re.sub(r'^(I want|show me)\s+', '', role_str, flags=re.IGNORECASE).strip()
            if role_str:
                roles = [role_str]
    else:
        parts = re.split(r'\s+with\s+', query_text, flags=re.IGNORECASE)
        if parts:
            role_str = parts[0].strip()
            role_str = re.sub(r'^(I want|show me)\s+', '', role_str, flags=re.IGNORECASE).strip()
            if role_str:
                roles = [role_str]

    # Extract location if provided (e.g., "in Bangalore")
    location_match = re.search(r'in\s+([A-Za-z ]+)', query_text, re.IGNORECASE)
    if location_match:
        location = location_match.group(1).strip()

    return roles, experience, location, salary_count

def generate_custom_query_post(filtered_df, roles, location, count=None):
    """
    Generates the custom query post text directly from filtered data.
    If a count is provided, select that many top entries; otherwise group by company.
    """
    if count is not None:
        sorted_df = filtered_df.sort_values("Salary", ascending=False)
        selected_df = sorted_df.head(count)
        companies = ", ".join([clean_company_name(comp) for comp in selected_df["Company"].unique()]) if "Company" in selected_df.columns else "Unknown Companies"
        n = len(selected_df)
        entries = [format_salary_entry_custom(row) for _, row in selected_df.iterrows()]
    else:
        if "Company" not in filtered_df.columns:
            selected_df = filtered_df.head(5)
            companies = "Unknown Companies"
        else:
            sorted_df = filtered_df.sort_values("Salary", ascending=False)
            selected_df = sorted_df.groupby("Company", as_index=False).first()
            companies = ", ".join([clean_company_name(comp) for comp in selected_df["Company"].unique()])
        n = len(selected_df)
        entries = [format_salary_entry_custom(row) for _, row in selected_df.iterrows()]

    role_str = roles[0] if roles else ""
    header_info = f"in {location}" if location else f"for {companies}"
    post = (f"{n} {role_str} Salaries {header_info} -\n" +
            "\n".join(entries) +
            "\nWe share 5 new salaries from Grapevine every 12 hours.\nHit follow")
    return post

async def analyze_custom_query(api_key, roles, location, filtered_df, count=None):
    return generate_custom_query_post(filtered_df, roles, location, count), None

async def analyze_salaries(api_key, company_name, cleaned_df):
    # Standard (viral) post generation using Gemini
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash-lite')
    viral_df = select_viral_entries(cleaned_df)
    formatted_entries = [format_salary_entry(row) for _, row in viral_df.iterrows()]
    data_text = "\n".join(formatted_entries)
    try:
        final_prompt = f"""Create a social media post with EXACTLY 5 salaries.
STRICTLY adhere to this EXACT format. DO NOT DEVIATE.

5 Salaries of {company_name}

{data_text}

We share 5 new salaries from Grapevine every 12 hours.
Hit follow"""
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(None, model.generate_content, final_prompt)
        return response.text, None
    except Exception as e:
        return None, str(e)

# ------------------ Standard (Viral Role) Feature ------------------

gemini_api_key = st.text_input("Enter your Gemini API key", type="password")
uploaded_files = st.file_uploader("Upload CSV(s)", type="csv", accept_multiple_files=True)

if uploaded_files and gemini_api_key:
    st.header("Standard Post Generation (Viral Role Selection)")
    if st.button("1M+ Users .. Lessgo!"):
        all_results = {}
        for uploaded_file in uploaded_files:
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file)
                default_company_name = uploaded_file.name.split('.')[0]
                cleaned_df = clean_data(df)
                with st.spinner(f"Generating post for {default_company_name}..."):
                    result, error = asyncio.run(analyze_salaries(gemini_api_key, default_company_name, cleaned_df))
                all_results[default_company_name] = (result, error)
            except Exception as e:
                all_results[uploaded_file.name.split('.')[0]] = (None, str(e))
        for company_name, (result, error) in all_results.items():
            if error:
                st.error(f"Error for {company_name}: {error}")
            else:
                st.subheader(f"Generated Post for {company_name}")
                st.code(result)
                st.download_button(
                    f"Download Post for {company_name}",
                    result,
                    file_name=f"{company_name.lower()}_post.txt"
                )
    
    # Individual File Processing:
    for uploaded_file in uploaded_files:
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file)
            default_company_name = uploaded_file.name.split('.')[0]
            company_name = st.text_input(f"Company Name for {uploaded_file.name}",
                                         default_company_name,
                                         key=f"company_name_{uploaded_file.name}")
            st.write(f"Raw Data Preview for {uploaded_file.name}:")
            st.dataframe(df.head())
            column_options = df.columns.tolist()
            manual_yoe_column = st.selectbox("Select YOE column (optional)",
                                             options=["None"] + column_options,
                                             key=f"yoe_select_{uploaded_file.name}")
            manual_yoe_column = manual_yoe_column if manual_yoe_column != "None" else None
            cleaned_df = clean_data(df, manual_yoe_column)
            st.success(f"Cleaned {len(cleaned_df)} rows for {uploaded_file.name}!")
            if st.button(f"Generate Post for {company_name}", key=f"generate_{uploaded_file.name}"):
                with st.spinner("Generating..."):
                    result, error = asyncio.run(analyze_salaries(gemini_api_key, company_name, cleaned_df))
                if error:
                    st.error(f"Error: {error}")
                else:
                    st.subheader(f"Generated Post for {company_name}")
                    st.code(result)
                    st.download_button(
                        f"Download Post ({company_name})",
                        result,
                        file_name=f"{company_name.lower()}_post.txt"
                    )
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")
elif uploaded_files:
    st.warning("Please enter your Gemini API key.")
elif gemini_api_key:
    st.info("Please upload at least one CSV file.")
else:
    st.info("Enter your Gemini API key and upload CSV files to begin.")

# ------------------ Custom Query Feature ------------------

st.markdown("---")
st.header("Custom Query Feature")
query_text = st.text_area("Enter your natural language query (e.g., 'Software Engineer with experience of 5 years and need 2 salaries'):")

if query_text and st.button("Run Query"):
    if uploaded_files:
        combined_data = []
        for uploaded_file in uploaded_files:
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file)
                default_company_name = uploaded_file.name.split('.')[0]
                # Add company info for custom query filtering/formatting
                df["Company"] = default_company_name
                cleaned_df = clean_data(df)
                combined_data.append(cleaned_df)
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name} in custom query: {e}")
        if combined_data:
            combined_df = pd.concat(combined_data, ignore_index=True)
            st.write("Combined Data from all files:", combined_df.head())
            roles, experience, location, salary_count = parse_query(query_text)
            if roles:
                role_mask = combined_df["Job_Title"].apply(lambda x: any(role.lower() in x.lower() for role in roles))
            else:
                role_mask = pd.Series([True] * len(combined_df), index=combined_df.index)
            if experience is not None:
                exp_mask = combined_df["YOE"].apply(lambda x: abs(x - experience) <= 1)
            else:
                exp_mask = pd.Series([True] * len(combined_df), index=combined_df.index)
            filtered_df = combined_df[role_mask & exp_mask]
            st.write("Filtered Data after applying query:", filtered_df.head())
            if filtered_df.empty:
                st.warning("No matching entries found for the query.")
            else:
                with st.spinner("Generating custom query post..."):
                    result, error = asyncio.run(analyze_custom_query(gemini_api_key, roles, location, filtered_df, salary_count))
                if error:
                    st.error(f"Error: {error}")
                else:
                    st.subheader("Custom Query Generated Post")
                    st.code(result)
                    st.download_button("Download Custom Query Post", result, file_name="custom_query_post.txt")
    else:
        st.warning("Please upload CSV files for the custom query feature.")
