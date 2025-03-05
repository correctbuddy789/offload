import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
import openai
import asyncio
import re
from sklearn.metrics.pairwise import cosine_similarity
import time

class SalaryInsightsApp:
    def __init__(self):
        """Initialize the Salary Insights App with default configurations."""
        # Synonyms for robust column matching
        self.YOE_SYNONYMS = ["yoe", "years", "experience"]
        self.TITLE_SYNONYMS = ["job_title", "title", "role", "position", "designation"]
        self.SALARY_SYNONYMS = ["total", "salary", "compensation", "ctc", "annual"]
        
        # Initialize session state
        self._initialize_session_state()

    def _initialize_session_state(self):
        """Initialize or reset session state variables."""
        session_keys = [
            'combined_df', 
            'processed_files', 
            'job_embeddings', 
            'job_descriptions'
        ]
        
        for key in session_keys:
            if key not in st.session_state:
                st.session_state[key] = None

    def find_best_column_match(self, columns_list, synonyms):
        """Find the best matching column based on synonyms."""
        for col in columns_list:
            col_lower = col.lower().strip()
            for syn in synonyms:
                if syn in col_lower:
                    return col
        return None

    def detect_and_rename_columns(self, df):
        """Detect and rename columns to standard format."""
        remaining_columns = list(df.columns)
        
        # Detect Year of Experience
        yoe_col = self.find_best_column_match(remaining_columns, self.YOE_SYNONYMS)
        if yoe_col:
            remaining_columns.remove(yoe_col)
        else:
            raise ValueError("Missing required column: Year of Experience")
        
        # Detect Job Title
        title_col = self.find_best_column_match(remaining_columns, self.TITLE_SYNONYMS)
        if title_col:
            remaining_columns.remove(title_col)
        else:
            raise ValueError("Missing required column: Job Title")
        
        # Detect Salary
        salary_col = self.find_best_column_match(remaining_columns, self.SALARY_SYNONYMS)
        if not salary_col:
            raise ValueError("Missing required column: Salary")
        remaining_columns.remove(salary_col)
        
        # Optionally detect Location column
        location_synonyms = ["location", "city", "area"]
        location_col = self.find_best_column_match(remaining_columns, location_synonyms)
        
        # Rename columns
        new_names = {
            yoe_col: "YOE", 
            title_col: "Job_Title", 
            salary_col: "Salary"
        }
        if location_col:
            new_names[location_col] = "Location"
        
        return df.rename(columns=new_names)

    def extract_yoe_from_string(self, job_title_string):
        """Extract years of experience from job title string."""
        if pd.isna(job_title_string):
            return None
        
        match = re.search(r'\((\d+\.?\d*)\s*yoe\)', job_title_string, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
        return None

    def clean_data(self, df, manual_yoe_column=None):
        """Clean and preprocess salary data."""
        if df.empty:
            raise ValueError("Uploaded CSV file is empty.")
        
        # Detect and rename columns
        df = self.detect_and_rename_columns(df)
        
        # Handle Years of Experience
        if manual_yoe_column and manual_yoe_column in df.columns:
            df['YOE'] = df[manual_yoe_column]
        elif df['YOE'].isna().all():
            extracted = df['Job_Title'].apply(self.extract_yoe_from_string)
            if extracted.notna().any():
                df['YOE'] = extracted
        
        # Convert salary to numeric
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
        
        # Drop rows with missing critical data
        df.dropna(subset=['Salary', 'Job_Title'], inplace=True)
        df['YOE'] = pd.to_numeric(df['YOE'], errors='coerce')
        df.dropna(subset=['YOE'], inplace=True)
        
        # Round YOE to whole number for display
        df['YOE'] = df['YOE'].round(0)
        
        return df

    def clean_company_name(self, company):
        """Clean company name by removing query result artifacts."""
        return re.sub(r'\s*-\s*query_result.*$', '', company, flags=re.IGNORECASE)

    def format_salary_entry(self, row):
        """Format salary entry for standard viral posts."""
        job_title = row['Job_Title']
        yoe = row['YOE']
        salary_in_lakhs = row['Salary'] / 100000
        
        if salary_in_lakhs >= 100:
            salary_str = f"{salary_in_lakhs / 100:.2f} Crores"
        else:
            salary_str = f"{salary_in_lakhs:.1f} Lakhs"
        
        return f"{job_title} with {int(yoe)} YOE earns {salary_str}"

    def format_salary_entry_custom(self, row):
        """Format salary entry for custom query posts."""
        job_title = re.sub(r'\s*-\s*query_result.*$', '', row['Job_Title'], flags=re.IGNORECASE)
        yoe = row['YOE']
        salary_in_lakhs = row['Salary'] / 100000
        company = row.get("Company", "")
        company = self.clean_company_name(company)
        
        if salary_in_lakhs >= 100:
            salary_str = f"{salary_in_lakhs / 100:.2f} Crores"
        else:
            salary_str = f"{int(salary_in_lakhs)} LPA"
        
        return f"{job_title} at {company} with {int(yoe)} YOE earns {salary_str}"

    def select_viral_entries(self, df):
        """Select top job titles for viral posts."""
        job_counts = df['Job_Title'].value_counts()
        top_titles = job_counts.head(5).index.tolist()
        selected_rows = []
        
        for title in top_titles:
            subset = df[df['Job_Title'] == title]
            rep_row = subset.sort_values('Salary', ascending=False).iloc[0]
            selected_rows.append(rep_row)
        
        return pd.DataFrame(selected_rows)

    @st.cache_data(ttl=3600)
    def get_embeddings_batch(self, texts, api_key):
        """Get embeddings for multiple texts in a single API call."""
        client = openai.OpenAI(api_key=api_key)
        try:
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=texts
            )
            return [item.embedding for item in response.data], None
        except Exception as e:
            return None, str(e)

    @st.cache_data(ttl=3600)
    def prepare_job_descriptions(self, df_json):
        """Prepare job descriptions for embedding."""
        df = pd.read_json(df_json)
        job_descriptions = []
        
        for _, row in df.iterrows():
            job_desc = f"{row['Job_Title']} with {int(row['YOE'])} YOE"
            if 'Location' in row and not pd.isna(row['Location']):
                job_desc += f" in {row['Location']}"
            job_descriptions.append(job_desc)
        
        return job_descriptions

    def find_matching_jobs(self, query, df, embeddings, openai_api_key, top_n=5):
        """Find matching jobs using cosine similarity."""
        start_time = time.time()
        
        # Get query embedding
        query_embeddings, error = self.get_embeddings_batch([query], openai_api_key)
        if error:
            return None, None, f"Error getting query embedding: {error}"
        
        query_embedding = query_embeddings[0]
        
        # Calculate similarities
        query_embedding_np = np.array(query_embedding)
        job_embeddings_np = np.array(embeddings)
        
        similarities = cosine_similarity([query_embedding_np], job_embeddings_np)[0]
        
        # Get top matching indices
        top_indices = np.argsort(similarities)[-top_n:][::-1]
        
        processing_time = time.time() - start_time
        
        return df.iloc[top_indices], similarities[top_indices], processing_time

    def generate_custom_query_post(self, filtered_df, query, location=None, count=None):
        """Generate custom query post text."""
        selected_df = filtered_df.head(count) if count is not None else filtered_df.head(5)
        
        companies = ", ".join([self.clean_company_name(comp) for comp in selected_df["Company"].unique()]) \
            if "Company" in selected_df.columns else "Unknown Companies"
        
        n = len(selected_df)
        entries = [self.format_salary_entry_custom(row) for _, row in selected_df.iterrows()]

        header_info = f"in {location}" if location else f"for {companies}"
        post = (f"{n} Salaries matching '{query}' {header_info} -\n" +
                "\n".join(entries) +
                "\nWe share 5 new salaries from Grapevine every 12 hours.\nHit follow")
        return post

    async def analyze_salaries(self, api_key, company_name, cleaned_df):
        """Analyze salaries and generate a viral post using Gemini."""
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash-lite')
        
        viral_df = self.select_viral_entries(cleaned_df)
        formatted_entries = [self.format_salary_entry(row) for _, row in viral_df.iterrows()]
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

    def run(self):
        """Main Streamlit application runner."""
        st.title("📊 obanai | Built at Grapevine")
        
        st.markdown("""
        ## About This App
        This app helps you generate engaging social media posts about salary data. It offers two modes:
        - *Standard Post Generation*: Creates viral-style posts using Gemini AI (requires Gemini API key)
        - *Custom Query Feature*: Uses OpenAI embeddings to semantically search for relevant salary data (requires OpenAI API key)
        """)
        
        tab1, tab2 = st.tabs(["Standard Post Generation", "Custom Query"])
        
        with tab1:
            self._standard_post_generation_tab()
        
        with tab2:
            self._custom_query_tab()

    def _standard_post_generation_tab(self):
        """Render the Standard Post Generation tab."""
        st.header("Standard Post Generation (Viral Role Selection)")
        gemini_api_key = st.text_input("Enter your Gemini API key", type="password", key="gemini_key")
        uploaded_files = st.file_uploader("Upload CSV(s)", type="csv", accept_multiple_files=True, key="standard_files")

        if uploaded_files and gemini_api_key:
            if st.button("1M+ Users .. Lessgo!"):
                all_results = {}
                for uploaded_file in uploaded_files:
                    try:
                        uploaded_file.seek(0)
                        df = pd.read_csv(uploaded_file)
                        default_company_name = uploaded_file.name.split('.')[0]
                        cleaned_df = self.clean_data(df)
                        with st.spinner(f"Generating post for {default_company_name}..."):
                            result, error = asyncio.run(self.analyze_salaries(gemini_api_key, default_company_name, cleaned_df))
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
            
            # Individual File Processing
            for uploaded_file in uploaded_files:
                try:
                    st.markdown("---")
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file)
                    default_company_name = uploaded_file.name.split('.')[0]
                    
                    company_name = st.text_input(
                        f"Company Name for {uploaded_file.name}",
                        default_company_name,
                        key=f"company_name_{uploaded_file.name}"
                    )
                    
                    st.write(f"Raw Data Preview for {uploaded_file.name}:")
                    st.dataframe(df.head())
                    
                    column_options = df.columns.tolist()
                    manual_yoe_column = st.selectbox(
                        "Select YOE column (optional)",
                        options=["None"] + column_options,
                        key=f"yoe_select_{uploaded_file.name}"
                    )
                    manual_yoe_column = manual_yoe_column if manual_yoe_column != "None" else None
                    
                    cleaned_df = self.clean_data(df, manual_yoe_column)
                    st.success(f"Cleane
