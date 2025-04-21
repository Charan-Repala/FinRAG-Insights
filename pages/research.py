import sys
import pysqlite3
sys.modules['sqlite3'] = pysqlite3
import os
import streamlit as st
import datetime
import re
from crewai import Agent, Task, LLM
from crewai_tools import SerperDevTool
from dotenv import load_dotenv
import google.generativeai as genai
import yfinance as yf
from fpdf import FPDF
from pydantic import BaseModel, Field
from typing import Type
from crewai.tools import BaseTool

# Load environment variables
load_dotenv()
GEMINI_API_KEY_TWO = os.environ.get('GEMINI_API_KEY_TWO')
SERPER_API_KEY_TWO = os.environ.get('SERPER_API_KEY_TWO')
genai.configure(api_key=GEMINI_API_KEY_TWO)

# Validate API keys
if not GEMINI_API_KEY_TWO:
    st.error("Gemini API key is not set. Please set GEMINI_API_KEY_TWO in your environment variables.")
    st.stop()
if not SERPER_API_KEY_TWO:
    st.error("Serper API key is not set. Please set SERPER_API_KEY_TWO in your environment variables.")
    st.stop()

# Initialize Gemini LLM
llm = LLM(
    model="gemini/gemini-2.0-flash-thinking-exp-01-21",
    temperature=0.5
)

# Initialize Serper tool
serper_tool = SerperDevTool()

# Define YFinanceHistoricalTool
class YFinanceHistoricalToolInput(BaseModel):
    symbol: str = Field(..., description="Stock symbol")
    start_year: int = Field(..., description="Start year")
    end_year: int = Field(..., description="End year")

class YFinanceHistoricalTool(BaseTool):
    name: str = "YFinanceHistoricalTool"
    description: str = "Fetch historical stock data from Yahoo Finance for a given period."
    args_schema: Type[BaseModel] = YFinanceHistoricalToolInput

    def _run(self, symbol: str, start_year: int, end_year: int) -> str:
        try:
            stock = yf.Ticker(symbol)
            start_date = f"{start_year}-01-01"
            current_year = datetime.date.today().year

            if end_year < current_year:
                end_date = f"{end_year}-12-31"
            else:
                end_date = (datetime.date.today() - datetime.timedelta(days=1)).strftime('%Y-%m-%d')

            hist = stock.history(start=start_date, end=end_date)
            if hist.empty:
                return f"No historical data found for {symbol} from {start_year} to {end_year}."
            
            first_close = hist['Close'].iloc[0]
            last_close = hist['Close'].iloc[-1]
            percent_change = (last_close - first_close) / first_close * 100
            max_close = hist['Close'].max()
            min_close = hist['Close'].min()
            
            summary = (
                f"Stock: {symbol}\n"
                f"Period: {start_year} to {end_year}\n"
                f"Start Close: {first_close:.2f}\n"
                f"End Close: {last_close:.2f}\n"
                f"Highest Close: {max_close:.2f}\n"
                f"Lowest Close: {min_close:.2f}\n"
                f"Change: {percent_change:.2f}%"
            )
            return summary
        except Exception as e:
            return f"Error fetching historical stock data: {str(e)}"

# Initialize tools
yfinance_tool = YFinanceHistoricalTool()

# Define Agents
research_agent = Agent(
    role="Company Researcher",
    goal="Gather stock data and news about the specified company for the given period.",
    backstory="You are an expert researcher skilled in analyzing companies using web tools and financial data.",
    verbose=False,
    tools=[serper_tool, yfinance_tool],
    llm=llm
)

use_case_agent = Agent(
    role="Data Analyst",
    goal="Analyze stock performance and news to identify trends and key events.",
    backstory="You are a data analyst who interprets financial data and news to provide insights.",
    verbose=False,
    llm=llm
)

report_agent = Agent(
    role="Report Compiler",
    goal="Compile the final report with structured sections in a PDF-friendly format.",
    backstory="You are a skilled writer who creates clear, actionable reports for stakeholders.",
    verbose=False,
    llm=llm
)

# Streamlit App
st.set_page_config(page_title="FinRAG-Insights Research Tool", page_icon="🤖")
st.title("FinRAG-Insights Research Tool🤖")

# Create proposals directory if it doesn’t exist
if not os.path.exists("proposals"):
    os.makedirs("proposals")

# List previous proposals (PDFs)
proposal_files = [f for f in os.listdir("proposals") if f.endswith(".pdf")]
proposal_files.sort(key=lambda x: os.path.getmtime(os.path.join("proposals", x)), reverse=True)

# Previous Proposals Section
st.subheader("Previous Proposals")
if proposal_files:
    selected_proposal = st.selectbox("Select a previous proposal", proposal_files)
    with open(os.path.join("proposals", selected_proposal), "rb") as f:
        st.download_button(
            label="Download Selected Proposal",
            data=f,
            file_name=selected_proposal,
            mime="application/pdf"
        )
    if st.button("Delete Selected Proposal"):
        os.remove(os.path.join("proposals", selected_proposal))
        st.success(f"Deleted {selected_proposal}")
        st.rerun()
else:
    st.info("No previous proposals found.")

# User Input
st.subheader("Company Research Input")
company_name = st.text_input("Enter the company name:", "")
stock_symbol = st.text_input("Enter the stock symbol (e.g., TSLA for Tesla):", "")
start_year = st.number_input("Start Year:", min_value=1900, max_value=2025, value=2023)
end_year = st.number_input("End Year:", min_value=1900, max_value=2025, value=2025)

# Validate start_year and end_year
if start_year > end_year:
    st.error("Start year cannot be after end year.")
    st.stop()

# Sanitize text function
def sanitize_text(text):
    replacements = {
        '\u2019': "'",
        '\u2018': "'",
        '\u2013': "-",
        '\u2014': "--",
        '\u2026': "..."
    }
    for original, replacement in replacements.items():
        text = text.replace(original, replacement)
    # Encode to cp1252, replacing any remaining unsupported characters with '?'
    return text.encode('cp1252', 'replace').decode('cp1252')

if st.button("Generate Proposal"):
    if company_name and stock_symbol:
        with st.spinner("Generating proposal..."):
            try:
                # Define research task 
                research_description = (
                    f"Conduct a comprehensive research on {company_name} (stock symbol: {stock_symbol}) for the period from {start_year} to {end_year}. "
                    "Utilize the YFinanceHistoricalTool to retrieve historical stock data, including opening and closing prices, volume, and any significant fluctuations. "
                    "Additionally, employ the Serper tool to gather news articles related to '{company_name} product launches' and '{company_name} market trends' within the specified years. "
                    "Provide a concise summary of the stock performance, highlighting key metrics such as overall percentage change, highest and lowest points, and any notable patterns. "
                    "Also, compile a list of the most relevant news articles, including their titles, brief descriptions, and URLs, focusing on major product launches and market trends that could impact the company's performance."
                )
                research_task = Task(
                    description=research_description,
                    expected_output="A summary of stock performance and a list of key news articles with titles, descriptions, and URLs.",
                    agent=research_agent
                )

                # Define analysis task 
                use_case_description = (
                    f"Analyze the collected data for {company_name} from {start_year} to {end_year}. "
                    "For the stock performance, calculate the overall percentage change, identify any significant spikes or drops, and determine the general trend (e.g., bullish, bearish, volatile). "
                    "Examine the news articles to extract information about major product launches, including the products' names, launch dates, and any reported reception or impact. "
                    "Also, identify key market trends mentioned in the news that could affect the company's industry or operations. "
                    "Synthesize this information to provide insights into how these events might have influenced or could influence the company's stock performance and market position."
                )
                use_case_task = Task(
                    description=use_case_description,
                    expected_output="A summary of stock trend and key events (product launches and market trends).",
                    agent=use_case_agent
                )

                # Define report task 
                report_description = (
                    f"Compile a comprehensive report for {company_name} covering {start_year} to {end_year}. "
                    "Structure the report with numbered sections:\n"
                    "1. Stock Performance: Include key metrics (start/end prices, percentage change, high/low points) and trend analysis.\n"
                    "2. Product Launches: List launches by year with brief details.\n"
                    "3. Market Trends: Summarize positive and negative trends with implications.\n"
                    "4. Conclusion: Integrate findings and provide an outlook.\n"
                    "Use clear, concise language suitable for a professional PDF."
                )
                report_task = Task(
                    description=report_description,
                    expected_output="A structured report with numbered sections separated by blank lines.",
                    agent=report_agent
                )

                # Execute tasks
                st.write(f"Agent: {research_agent.role} is gathering data on {company_name}.")
                research_output = research_task.execute_sync(agent=research_agent)
                
                st.write(f"Agent: {use_case_agent.role} is analyzing the data.")
                use_case_output = use_case_task.execute_sync(agent=use_case_agent, context=research_output.raw)
                
                st.write(f"Agent: {report_agent.role} is compiling the final report.")
                # Concatenate outputs into a single string for context
                context_for_report = research_output.raw + '\n\n' + use_case_output.raw
                report_output = report_task.execute_sync(agent=report_agent, context=context_for_report)

                # Get the final report text
                report_text = report_output.raw  

                # Sanitize the report text 
                sanitized_report_text = sanitize_text(report_text)

                # Generate PDF 
                pdf = FPDF()
                pdf.add_page()

                # Title Page
                pdf.set_font("Helvetica", 'B', 20)
                pdf.cell(0, 10, txt=f"Report for {company_name}", ln=1, align='C')
                pdf.set_font("Helvetica", size=16)
                pdf.cell(0, 10, txt=f"({start_year}-{end_year})", ln=1, align='C')
                pdf.ln(10)
                pdf.set_font("Helvetica", size=12)
                pdf.multi_cell(0, 10, txt="This report analyzes stock performance, product launches, and market trends for the specified period.")
                pdf.ln(10)

                sections = sanitized_report_text.split('\n\n')
                section_number = 1
                for section in sections:
                    if ':' in section:
                        title, content = section.split(':', 1)
                        title = title.strip().replace('**', '').lstrip('- ').strip()  
                        numbered_title = f"{section_number}. {title}"
                        
                        # Heading
                        pdf.set_font("Helvetica", 'B', 14)
                        pdf.cell(0, 10, txt=numbered_title, ln=1)
                        
                        # Body text
                        pdf.set_font("Helvetica", size=12)
                        pdf.multi_cell(0, 10, txt=content.strip())
                        
                        # Add chart placeholder 
                        if "Stock Performance" in title:
                            pdf.ln(5)
                            pdf.set_font("Helvetica", 'I', 12)
                            pdf.cell(0, 10, txt="[Insert Stock Performance Chart Sanctions Here]", ln=1)
                        
                        pdf.ln(10)
                        section_number += 1
                    else:
                        pdf.multi_cell(0, 10, txt=section.strip())
                        pdf.ln(5)

                # Save PDF
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                pdf_file = f"{company_name}_{start_year}_{end_year}_{timestamp}.pdf"
                pdf_file_path = os.path.join("proposals", pdf_file)
                pdf.output(pdf_file_path)

                # Provide download button
                with open(pdf_file_path, "rb") as f:
                    st.download_button(
                        label="Download PDF Proposal",
                        data=f,
                        file_name=pdf_file,
                        mime="application/pdf"
                    )

                # Display success message
                st.write("✅ **Proposal generated successfully!**")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.error("Please enter both company name and stock symbol.")

if __name__ == "__main__":
    pass
