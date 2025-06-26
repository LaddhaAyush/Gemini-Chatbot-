# from dotenv import load_dotenv
# load_dotenv()

# import streamlit as st
# import os
# import sqlite3

# from langchain_groq import ChatGroq
# from langchain_core.prompts import ChatPromptTemplate

# ## Configure ChatGroq LLM
# llm = ChatGroq(
#     groq_api_key=os.getenv("GROQ_API_KEY"),
#     model_name="gemma2-9b-it"
# )

# ## Function To Load LLM and get SQL query as response
# def get_groq_sql_response(question, prompt):
#     full_prompt = prompt[0] + "\nQuestion: " + question
#     response = llm.invoke(full_prompt)
#     return response.content.strip()

# ## Function To retrieve query from the database
# def read_sql_query(sql, db):
#     conn = sqlite3.connect(db)
#     cur = conn.cursor()
#     cur.execute(sql)
#     rows = cur.fetchall()
#     conn.commit()
#     conn.close()
#     return rows

# ## Define Your Prompt
# prompt = [
#     """
#     You are an expert in converting English questions to SQL queries.
#     The SQL database has the name STUDENT and has the following columns - NAME, CLASS, SECTION.

#     Example 1:
#     Q: How many entries of records are present?
#     A: SELECT COUNT(*) FROM STUDENT;

#     Example 2:
#     Q: Tell me all the students studying in Data Science class?
#     A: SELECT * FROM STUDENT WHERE CLASS="Data Science";

#     IMPORTANT:
#     - Do not include triple backticks.
#     - Do not prefix the SQL query with the word 'sql'.
#     """
# ]

# ## Streamlit App
# st.set_page_config(page_title="ChatGroq SQL Generator")
# st.header("ChatGroq-powered App To Retrieve SQL Data")

# question = st.text_input("Enter your question in English:", key="input")

# submit = st.button("Submit")

# if submit:
#     sql_query = get_groq_sql_response(question, prompt)
#     st.subheader("Generated SQL Query")
#     st.code(sql_query, language="sql")

#     try:
#         result = read_sql_query(sql_query, "student.db")
#         st.subheader("Query Result")
#         for row in result:
#             st.write(row)
#     except Exception as e:
#         st.error(f"Error running SQL: {e}")

# Step-by-step improved version of your ChatGroq SQL Agent App

import os
import sqlite3
import streamlit as st
import re
from datetime import datetime

from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


def get_sql_query(user_query):
    groq_sys_prompt = ChatPromptTemplate.from_template("""
                    You are an expert in converting English questions to SQL query!
                    The SQL database has the name STUDENT and has the following columns - NAME, COURSE, 
                    SECTION and MARKS. 
                    
                    You can perform ALL types of SQL operations:
                    
                    SELECT Examples:
                    - "How many entries of records are present?" -> SELECT COUNT(*) FROM STUDENT;
                    - "Tell me all students studying in Data Science?" -> SELECT * FROM STUDENT WHERE COURSE='Data Science';
                    - "Show students with marks above 80" -> SELECT * FROM STUDENT WHERE MARKS > 80;
                    - "What is the average marks?" -> SELECT AVG(MARKS) FROM STUDENT;
                    
                    INSERT Examples:
                    - "Add a new student John in Data Science section A with 85 marks" -> INSERT INTO STUDENT (NAME, COURSE, SECTION, MARKS) VALUES ('John', 'Data Science', 'A', 85);
                    - "Insert student Mary in Computer Science section B with 92 marks" -> INSERT INTO STUDENT (NAME, COURSE, SECTION, MARKS) VALUES ('Mary', 'Computer Science', 'B', 92);
                    
                    UPDATE Examples:
                    - "Update marks to 95 for student John" -> UPDATE STUDENT SET MARKS = 95 WHERE NAME = 'John';
                    - "Change course to AI for all students in section A" -> UPDATE STUDENT SET COURSE = 'AI' WHERE SECTION = 'A';
                    - "Increase marks by 5 for all Data Science students" -> UPDATE STUDENT SET MARKS = MARKS + 5 WHERE COURSE = 'Data Science';
                    
                    DELETE Examples:
                    - "Delete student John" -> DELETE FROM STUDENT WHERE NAME = 'John';
                    - "Remove all students with marks below 50" -> DELETE FROM STUDENT WHERE MARKS < 50;
                    - "Delete all students from section C" -> DELETE FROM STUDENT WHERE SECTION = 'C';
                    
                    CREATE TABLE Examples:
                    - "Create the student table" -> CREATE TABLE IF NOT EXISTS STUDENT (NAME TEXT, COURSE TEXT, SECTION TEXT, MARKS INTEGER);
                    
                    DROP Examples:
                    - "Drop the student table" -> DROP TABLE IF EXISTS STUDENT;
                    
                    Important Rules:
                    - Return ONLY the SQL query, no explanations, no markdown formatting
                    - Do not include ``` or 'sql' in the output
                    - Use single quotes for string values in SQL
                    - For case-insensitive matching, use UPPER() or LOWER() functions
                    
                    Convert this English question to SQL: {user_query}
                                                       """)
    
    model = "llama3-8b-8192"
    llm = ChatGroq(
        groq_api_key=os.environ.get("GROQ_API_KEY"),
        model_name=model,
        verbose=True
    )

    chain = groq_sys_prompt | llm | StrOutputParser()
    response = chain.invoke({"user_query": user_query})
    return response.strip()


def execute_sql_with_smart_response(sql_query, original_query):
    """Execute SQL query and provide intelligent responses based on operation type"""
    database = "student.db"
    
    # Clean and analyze the SQL query
    sql_clean = sql_query.strip().rstrip(';')
    query_type = sql_clean.upper().split()[0]
    
    try:
        with sqlite3.connect(database) as conn:
            cursor = conn.cursor()
            
            if query_type == "SELECT":
                result = cursor.execute(sql_query).fetchall()
                
                # Get column names for better display
                column_names = [description[0] for description in cursor.description]
                
                if result:
                    response = {
                        "type": "select",
                        "success": True,
                        "data": result,
                        "columns": column_names,
                        "count": len(result),
                        "message": f"Found {len(result)} record(s) matching your query."
                    }
                else:
                    response = {
                        "type": "select",
                        "success": True,
                        "data": [],
                        "columns": column_names,
                        "count": 0,
                        "message": "No records found matching your criteria."
                    }
                
            elif query_type == "INSERT":
                cursor.execute(sql_query)
                conn.commit()
                affected_rows = cursor.rowcount
                
                # Get the inserted data for confirmation
                if "VALUES" in sql_query.upper():
                    # Extract values from INSERT statement for display
                    values_match = re.search(r"VALUES\s*\((.*?)\)", sql_query, re.IGNORECASE)
                    if values_match:
                        values = values_match.group(1).replace("'", "").replace('"', '')
                
                response = {
                    "type": "insert",
                    "success": True,
                    "affected_rows": affected_rows,
                    "message": f"✅ Successfully inserted {affected_rows} new record(s) into the database!",
                    "details": f"Operation: {original_query}"
                }
                
            elif query_type == "UPDATE":
                # Get records before update for comparison
                table_match = re.search(r"UPDATE\s+(\w+)", sql_query, re.IGNORECASE)
                where_match = re.search(r"WHERE\s+(.+)", sql_query, re.IGNORECASE)
                
                before_count = 0
                if table_match and where_match:
                    table_name = table_match.group(1)
                    where_clause = where_match.group(1)
                    try:
                        before_result = cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE {where_clause}").fetchone()
                        before_count = before_result[0] if before_result else 0
                    except:
                        pass
                
                cursor.execute(sql_query)
                conn.commit()
                affected_rows = cursor.rowcount
                
                response = {
                    "type": "update",
                    "success": True,
                    "affected_rows": affected_rows,
                    "message": f"✅ Successfully updated {affected_rows} record(s) in the database!",
                    "details": f"Operation: {original_query}"
                }
                
            elif query_type == "DELETE":
                # Get count before deletion
                table_match = re.search(r"FROM\s+(\w+)", sql_query, re.IGNORECASE)
                where_match = re.search(r"WHERE\s+(.+)", sql_query, re.IGNORECASE)
                
                before_count = 0
                if table_match:
                    table_name = table_match.group(1)
                    if where_match:
                        where_clause = where_match.group(1)
                        try:
                            before_result = cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE {where_clause}").fetchone()
                            before_count = before_result[0] if before_result else 0
                        except:
                            pass
                
                cursor.execute(sql_query)
                conn.commit()
                affected_rows = cursor.rowcount
                
                response = {
                    "type": "delete",
                    "success": True,
                    "affected_rows": affected_rows,
                    "message": f"✅ Successfully deleted {affected_rows} record(s) from the database!",
                    "details": f"Operation: {original_query}",
                    "warning": "⚠️ This operation cannot be undone!" if affected_rows > 0 else ""
                }
                
            elif query_type in ["CREATE", "DROP", "ALTER"]:
                cursor.execute(sql_query)
                conn.commit()
                
                operation_name = "created" if query_type == "CREATE" else "dropped" if query_type == "DROP" else "altered"
                
                response = {
                    "type": "ddl",
                    "success": True,
                    "message": f"✅ Successfully {operation_name} database structure!",
                    "details": f"Operation: {original_query}",
                    "sql_executed": sql_query
                }
                
            else:
                response = {
                    "type": "unknown",
                    "success": False,
                    "message": f"❌ Unsupported SQL operation: {query_type}",
                    "sql_query": sql_query
                }
                
    except sqlite3.Error as e:
        response = {
            "type": "error",
            "success": False,
            "message": f"❌ Database Error: {str(e)}",
            "sql_query": sql_query,
            "original_query": original_query
        }
    except Exception as e:
        response = {
            "type": "error",
            "success": False,
            "message": f"❌ Unexpected Error: {str(e)}",
            "sql_query": sql_query
        }
    
    return response


def initialize_database():
    """Initialize the database with sample data"""
    database = "student.db"
    
    try:
        with sqlite3.connect(database) as conn:
            cursor = conn.cursor()
            
            # Create table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS STUDENT (
                    NAME TEXT NOT NULL,
                    COURSE TEXT NOT NULL,
                    SECTION TEXT NOT NULL,
                    MARKS INTEGER NOT NULL
                )
            """)
            
            # Check if table is empty
            cursor.execute("SELECT COUNT(*) FROM STUDENT")
            count = cursor.fetchone()[0]
            
            if count == 0:
                sample_data = [
                    ('Alice Johnson', 'Data Science', 'A', 85),
                    ('Bob Smith', 'Computer Science', 'B', 78),
                    ('Charlie Brown', 'Data Science', 'A', 92),
                    ('Diana Prince', 'Mathematics', 'C', 88),
                    ('Eve Wilson', 'Computer Science', 'B', 76),
                    ('Frank Miller', 'Physics', 'A', 82),
                    ('Grace Lee', 'Data Science', 'C', 94),
                    ('Henry Davis', 'Mathematics', 'B', 71)
                ]
                
                cursor.executemany("INSERT INTO STUDENT VALUES (?, ?, ?, ?)", sample_data)
                conn.commit()
                
                return {
                    "success": True,
                    "message": f"✅ Database initialized successfully with {len(sample_data)} sample records!",
                    "records_added": len(sample_data)
                }
            else:
                return {
                    "success": True,
                    "message": f"ℹ️ Database already exists with {count} records.",
                    "existing_records": count
                }
                
    except Exception as e:
        return {
            "success": False,
            "message": f"❌ Error initializing database: {str(e)}"
        }


def display_results(response):
    """Display results based on response type with rich formatting"""
    
    if response["type"] == "select":
        st.success(response["message"])
        
        if response["data"]:
            # Create a formatted table display
            st.subheader("📊 Query Results")
            
            # Display as a proper table
            import pandas as pd
            df = pd.DataFrame(response["data"], columns=response["columns"])
            st.dataframe(df, use_container_width=True)
            
            # Additional stats for numeric results
            if len(response["columns"]) == 1 and len(response["data"]) == 1:
                # Single value result (like COUNT, AVG, etc.)
                value = response["data"][0][0]
                st.metric(label=response["columns"][0], value=value)
        else:
            st.info("No data found matching your criteria.")
    
    elif response["type"] in ["insert", "update", "delete"]:
        st.success(response["message"])
        st.info(f"📝 {response['details']}")
        
        if response.get("warning"):
            st.warning(response["warning"])
        
        # Show affected rows count
        if response["affected_rows"] > 0:
            st.metric("Affected Rows", response["affected_rows"])
    
    elif response["type"] == "ddl":
        st.success(response["message"])
        st.info(f"📝 {response['details']}")
        with st.expander("SQL Executed"):
            st.code(response["sql_executed"], language="sql")
    
    elif response["type"] == "error":
        st.error(response["message"])
        with st.expander("Debug Information"):
            st.code(f"Original Query: {response.get('original_query', 'N/A')}")
            st.code(f"Generated SQL: {response.get('sql_query', 'N/A')}", language="sql")


def main():
    st.set_page_config(
        page_title="Smart Text-to-SQL Interface",
        page_icon="🗄️",
        layout="wide"
    )
    
    st.title("🗄️ Smart Text-to-SQL Interface")
    st.markdown("*Convert natural language to SQL and get intelligent responses!*")
    
    # Sidebar for database operations
    with st.sidebar:
        st.header("🔧 Database Operations")
        
        if st.button("🚀 Initialize Database", type="primary"):
            init_result = initialize_database()
            if init_result["success"]:
                st.success(init_result["message"])
            else:
                st.error(init_result["message"])
        
        if st.button("👀 View All Records"):
            result = execute_sql_with_smart_response("SELECT * FROM STUDENT", "Show all records")
            display_results(result)
        
        if st.button("📊 Database Stats"):
            stats_result = execute_sql_with_smart_response(
                "SELECT COUNT(*) as Total_Students, AVG(MARKS) as Average_Marks, MAX(MARKS) as Highest_Marks, MIN(MARKS) as Lowest_Marks FROM STUDENT", 
                "Show database statistics"
            )
            display_results(stats_result)
    
    # Main interface
    st.markdown("### 💬 Natural Language Query")
    
    # Examples section
    with st.expander("📚 Example Queries"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📊 SELECT Operations:**")
            st.markdown("- Show all students")
            st.markdown("- How many students are in Data Science?")  
            st.markdown("- Students with marks above 80")
            st.markdown("- Average marks by course")
            
            st.markdown("**➕ INSERT Operations:**")
            st.markdown("- Add student John in AI section A with 90 marks")
            st.markdown("- Insert new student Sarah studying Physics")
        
        with col2:
            st.markdown("**✏️ UPDATE Operations:**")
            st.markdown("- Update marks to 95 for Alice Johnson")
            st.markdown("- Change all section A students to section B")
            st.markdown("- Increase marks by 5 for Computer Science students")
            
            st.markdown("**🗑️ DELETE Operations:**")
            st.markdown("- Delete student Bob Smith")
            st.markdown("- Remove students with marks below 60")
    
    # Query input
    user_query = st.text_area(
        "Enter your query in plain English:",
        placeholder="e.g., Show me all students in Data Science with marks above 85",
        height=100
    )
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        execute_btn = st.button("🚀 Execute Query", type="primary")
    
    with col2:
        if st.button("🧹 Clear"):
            st.rerun()
    
    # Query execution
    if execute_btn and user_query.strip():
        with st.spinner("🔄 Processing your query..."):
            # Generate SQL
            sql_query = get_sql_query(user_query)
            
            # Display generated SQL
            st.subheader("🔍 Generated SQL Query")
            st.code(sql_query, language="sql")
            
            # Execute and get smart response
            result = execute_sql_with_smart_response(sql_query, user_query)
            
            # Display results
            st.subheader("📋 Results")
            display_results(result)
            
            # Log the operation
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with st.expander("📝 Operation Log"):
                st.json({
                    "timestamp": timestamp,
                    "user_query": user_query,
                    "generated_sql": sql_query,
                    "operation_type": result["type"],
                    "success": result["success"]
                })
    
    elif execute_btn:
        st.warning("⚠️ Please enter a query first!")


if __name__ == '__main__':
    main()