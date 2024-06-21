#%pip install streamlit openai llama-index nltk

import os
import time
import fnmatch
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine
from openai import OpenAI
from pathlib import Path

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")


# Load environment variables
load_dotenv()
db_path = '/main/excel_data.db'

my_file = Path(db_path)
if my_file.is_file():
    os.remove(db_path)
    print(db_path)

#Dictionary to store the extracted dataframes
data = {}
engine = create_engine('sqlite:///excel_data.db', echo=True)


def main():

    client = OpenAI()
    st.set_page_config(page_title='DataAnalysis', page_icon='	:graph:', layout='wide' )
    st.title("Data Analysis for supplychain")
    st.markdown('<style>div.block-container {paddin-top:1rem;}</style>', unsafe_allow_html=True)
    
    #Side Menu Bar
    with st.sidebar:
        st.title("")
        #Activating Demo Data
        st.markdown("Upload the file ")
        file_upload = st.file_uploader("",accept_multiple_files=True,type = ['csv','xls','xlsx'])
        print(file_upload)


        # #Adding users API Key
        # user_api_key =  "sk-proj-qP1ZQPrJc8ZaQ9uz8Vu8T3BlbkFJYzRYiQB5QM8aGgbmgNSD"


    if len(file_upload) != 0:
        print(file_upload)
        for i in file_upload:
            filename = i.name

        data  = extract_file_csv(file_upload, engine)
        st.write('\nThe file has been uploaded successfully.')
       
        df1 = st.selectbox("Here's your uploaded data!",
                          tuple(data.keys()),index=0)
        st.dataframe(data[df1])

        db = SQLDatabase.from_uri('sqlite:///excel_data.db')
       
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True)
            
        #starting the chat with the PandasAI agent
        chat_window(agent_executor, client)
        
    else:
        st.warning("Please select the correct file. You can upload a CSV or an Excel file.")

   
#Functuion for chat window
def chat_window( analyst, client):
    with st.chat_message("assistant"):
        st.markdown("Enter your queries: ")

    #Initilizing message history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    

    #Displaying the message history on re-reun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            #priting the questions
            if 'question' in message:
                st.markdown(message["question"])
            #printing the code generated and the evaluated code
            elif 'response' in message:
                #getting the response
                if message['chart_path'] is not None:
                    st.image(message['chart_path'])
                st.markdown(message['response'])
                
            #retrieving error messages
            elif 'error' in message:
                st.text(message['error'])


    #Getting the questions from the users
    from streamlit_pills import pills
    selected =None
    st.session_state.pills_index = None
    selected = pills("Sample Queries", [ "None", "Display a pie chart to illustrate the product distribution.", "Analyze the sales trend using a line chart."])

    user_question = st.chat_input("What are you curious about? ")
    

    # prompt=''' 
    
    # You are an expert management consultant specializing in supply chain management. 
    # And ensure the insights generated is directly related to the graph generated.
    # Analyze the given dataset to answer the given prompt :{0}. 

    # Also remove the duplicate entry id any. Use only the provided data for analysis.
    # '''


    prompt = '''
        <prompt>
        <context>
            You are an expert management consultant specializing in supply chain management. You need to analyze the given dataset to answer specific questions related to supply chain performance. The dataset may contain duplicate entries, which need to be identified and removed before analysis.
        </context>
        <task>
            <steps>
            <step>Remove any duplicate entries from the provided dataset.</step>
            <step>Based on the query, Analyze the cleaned dataset to generate insights.</step>
            <step>Ensure the insights generated are directly related to the query given and graph generated.</step>
            </steps>
        </task>
        <query>
        {0}
        </query>

        <expectedOutput>
            <insights>
            <insight>Clear and concise analysis related to supply chain management.</insight>
            <insight>Specific insights that are directly supported by the graph generated from the dataset.</insight>
            </insights>
        </expectedOutput>
        </prompt>
        '''
    
    if selected != 'None':
        user_question = selected
        st.session_state.pills_index = None
        selected = 'None'

    query = prompt.format(user_question)

    if user_question:
        #Displaying the user question in the chat message
        with st.chat_message("user"):
            st.markdown(user_question)
        #Adding user question to chat history
        st.session_state.messages.append({"role":"user","question":user_question})
        
        try:
            with st.spinner("Analyzing..."):
                response  = analyst.invoke(query) # to extract the data from SQL db
                combined_string=str()
                chart_path, chart_content = generate_chart(response['output'], client)  #using assiastant api : we are generating charts
                insights = generate_insight(response['output'], client) #using assiastant api: we are generating insights


                if chart_path is not None:
                    my_file = Path(chart_path)
                    if my_file.is_file():
                        st.image(chart_path)
                    else:
                        chart_path = None
                    #    st.write(chart_content)

                if chart_path is not None and chart_content is not None :
                    combined_string += 'Chart Explanation: \n'+ chart_content

                if insights is not None:   
                    combined_string += ' \n'+ insights
                
                st.write(combined_string)
                st.session_state.messages.append({"role":"assistant","response":combined_string, "chart_path":chart_path})
        except Exception as e:
            st.write(e)
            error_message = "Sorry, Couldn't generate the answer! Please try rephrasing your question!"

        
    #Function to clear history
    def clear_chat_history():
        st.session_state.messages = []
    #Button for clearing history

    st.sidebar.markdown("Click to Clear Chat history")
    st.sidebar.button("Clear Chat",on_click=clear_chat_history)


def extract_file_csv(raw_files_list, engine):
    """
    This function extracts dataframes from the uploaded file/files
    Args: 
        raw_file: Upload_File object
    Processing: Based on the type of file read_csv or read_excel to extract the dataframes
    Output: 
        dfs:  a dictionary with the dataframes
    """

    dfs = {}
    for raw_file in raw_files_list:
        if raw_file.name.split('.')[1] == 'csv':
            csv_name = raw_file.name.split('.')[0]
            df = pd.read_csv(raw_file)
            dfs[csv_name] = df
            dfs[csv_name].to_sql(csv_name, con=engine, if_exists='replace', index=False)

        elif (raw_file.name.split('.')[1] == 'xlsx') or (raw_file.name.split('.')[1] == 'xls') :
            # Read the Excel file
            xls = pd.ExcelFile(raw_file)

            # Iterate through each sheet in the Excel file and store them into dataframes
            
            for sheet_name in xls.sheet_names:
                dfs[str(raw_file.name.split('.')[0])+"__"+sheet_name] = (pd.read_excel(raw_file, sheet_name=sheet_name))
                (pd.read_excel(raw_file, sheet_name=sheet_name)).to_sql(sheet_name, con=engine, if_exists='replace', index=False)

    #return the dataframes
    
    return dfs

def generate_chart(message, client):
    # Prepare the prompt

    i= len(fnmatch.filter(os.listdir('./main/exports/charts'), '*.png'))
    chart_path='./main/exports/charts/chart'+ '__'+str(i)+'.png'
    
    ci_prompt = "Please generate a chart using following data:  \n" + message

    # prompt ='''
    #     <prompt>
    #     <context>
    #         You are required to generate a chart using the provided dataset. Ensure the chart is neatly labelled and utilizes different colors to distinguish between various data points. A legend must be included to label the different elements within the chart for clear interpretation.
    #     </context>
    #     <task>
    #         <steps>
    #         <step>Based on the given data.</step>
    #         <step>Clean the dataset by removing any duplicate entries.</step>
    #         <step>Generate a chart using the cleaned dataset.</step>
    #         <step>Apply different colors to distinguish between various data points.</step>
    #         <step>Include a legend to label the different elements within the chart.</step>
    #         <step>Ensure all axes and data points are neatly labelled for clarity.</step>
    #         </steps>
    #     </task>

    #     <data>
    #     {0}
    #     </data>
    #     <expectedOutput>
    #         <chart>
    #         <description>A neatly labelled chart with different colors used for various data points and a legend included for clear labelling of the chart elements.</description>
    #         </chart>
    #     </expectedOutput>
    #     </prompt>
    #     '''

    # ci_prompt = prompt.format(message)
    # print(ci_prompt)

    try:
        # Create a thread and run the assistant
        thread = client.beta.threads.create(
            messages=[
                {
                    "role": "user",
                    "content": ci_prompt,
                }
            ]
        )

        assistant= client.beta.assistants.create(
                name = 'graph-generator',
                instructions="You generate plots for the given data",
                model="gpt-4o",
                tools=[{"type": "code_interpreter"}],
            )
        
        # Run the thread
        run = client.beta.threads.runs.create(
            assistant_id=assistant.id, thread_id=thread.id
        )

        # Poll the run status until it is completed
        while True:
            # Refresh the run to get the latest status
            run = client.beta.threads.runs.retrieve(
                run_id=run.id, thread_id=thread.id
            )

            if run.status == "completed":
                print("Generated chart, Run finished")

                # Get list of messages in the thread
                messages = client.beta.threads.messages.list(
                    thread_id=thread.id
                )

                # Get the latest message in the thread and retrieve file id
                print(messages.data[0])
                image_file_id = messages.data[0].content[0].image_file.file_id
                content_description = messages.data[0].content[1].text.value

                # Get the raw response from the file id
                raw_response = client.files.with_raw_response.content(
                    file_id=image_file_id
                )

                # Delete generated file
                client.files.delete(image_file_id)

                # Save the generated chart to a file
                with open(chart_path, "wb") as f:
                    f.write(raw_response.content)
                return (chart_path, content_description)

            elif run.status == "failed":
                print("Unable to generate chart")
                break

            # Wait for a short period before polling again to avoid hitting rate limits
            time.sleep(1)
            
    except Exception as e:
        print(e)

    return (None, "🤔 Could not analyse the query")




def generate_insight(message, client):
    # Prepare the prompt

    ci_prompt = "Please generate insights and next immediate action steps to improve, based on the following data: \n" + message

    try:
        # Create a thread and run the assistant
        thread1 = client.beta.threads.create(
            messages=[
                {
                    "role": "user",
                    "content": ci_prompt,
                }
            ]
        )

        assistant1= client.beta.assistants.create(
                name = 'insight-generator',
                instructions="You generate insights and next steps for the given data",
                model="gpt-4o",
                tools=[{"type": "code_interpreter"}],

            )
        
        # Run the thread
        run = client.beta.threads.runs.create(
            assistant_id=assistant1.id, thread_id=thread1.id
        )

        # Poll the run status until it is completed
        while True:
            # Refresh the run to get the latest status
            run = client.beta.threads.runs.retrieve(
                run_id=run.id, thread_id=thread1.id
            )

            if run.status == "completed":
                print("\nGenerated insights, Run finished")

                # Get list of messages in the thread
                messages = client.beta.threads.messages.list(
                    thread_id=thread1.id
                )

                # Get the latest message in the thread and retrieve file id
                #print(messages.data[0].content)
                import openai
                for i in range (0,len(messages.data[0].content)):
                    if (type(messages.data[0].content[i])==openai.types.beta.threads.text_content_block.TextContentBlock):
                            msg= messages.data[0].content[i].text.value
                            break
                    else:
                        msg=None
                    
                if msg is not None:
                    cleaned_msg = msg.replace('### ', '').replace('**', '')

                    # Printing the cleaned text
                    print(cleaned_msg)
                    return (cleaned_msg)
                else:
                    return None

            elif run.status == "failed":
                print("Unable to generate insights")
                break

            # Wait for a short period before polling again to avoid hitting rate limits
            time.sleep(1)
            
    except Exception as e:
        print(e)

    return (None, "🤔 Could you please rephrase your query and try again?")


if __name__ == "__main__":
    main()





