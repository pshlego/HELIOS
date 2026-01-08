end_to_end_qa_prompt = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>
You are given tables and their associated passages. Use the information from both the tables and the passages to answer the question concisely. The answer should be a short span, containing only a few words.{sensitive_keywords_instruction}

/*
{table_and_passages}
*/

Question: {question}
Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

few_shot_qa_prompt = """
<|start_header_id|>user<|end_header_id|>
Here is the table and passages to answer this question. Answer the question with a short span. The answer needs to be just in a few words.{sensitive_keywords_instruction}
/*
Title : List of politicians, lawyers, and civil servants educated at Jesus College, Oxford
col : Name | M | G | Degree | Notes
row 1 : Lalith Athulathmudali | 1955 | 1960 | BA Jurisprudence ( 2nd , 1958 ) , BCL ( 2nd , 1960 ) | President of the Oxford Union ( 1958 ) ; a Sri Lankan politician ; killed by the Tamil Tigers in 1993
row 2 : Neal Blewett ( HF ) | 1957 | 1959 | BA PPE ( 2nd ) | Member of the Australian House of Representatives ( 1977-1994 ) , Government Minister ( 1983-1994 ) , High Commissioner to the UK ( 1994-1998 )
row 3 : Joseph Clearihue | 1911 | 1914 | BA Jurisprudence ( 2nd , 1913 ) , BCL ( 3rd , 1914 ) | Canadian Rhodes scholar ; later became a member of the Legislative Assembly of British Columbia and a county court judge ; also chairman of the council of Victoria College , British Columbia ( which became the University of Victoria under his leadership )


Passages linked to row 1:
- Lalith Athulathmudali Lalith William Samarasekera Athulathmudali , PC ( Sinhala : ලලිත් ඇතුලත්මුදලි ; 26 November 1936 - 23 April 1993 ) , known as Lalith Athulathmudali , was Sri Lankan statesman . He was a prominent member of the United National Party , who served as Minister of Trade and Shipping ; Minister National Security and Deputy Minister of Defence ; Minister of Agriculture , Food and Cooperatives and finally Minister of Education . 
- Law degree A law degree is an academic degree conferred for studies in law . Such degrees are generally preparation for legal careers ; but while their curricula may be reviewed by legal authority , they do not themselves confer a license . A legal license is granted ( typically by examination ) and exercised locally ; while the law degree can have local , international , and world-wide aspects .
Passages linked to row 3:
- Joseph Clearihue Joseph Badenoch Clearihue ( 1887-1976 ) was a Canadian lawyer , judge , academic and politician .
- University of Victoria The University of Victoria ( UVic ) is a public comprehensive research university located in the Greater Victoria municipalities of Oak Bay and Saanich , British Columbia , Canada . The university traces its roots to Victoria College , an institution established in 1903 .
*/
Question: what is the full name of the Jesus College alumni who graduated in 1960 ?
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
The answer is: Lalith William Samarasekera Athulathmudali.
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Here is the table and passages to answer this question. Answer the question with a short span. The answer needs to be just in a few words.
/*
table caption : List of Rangers F.C. records and statistics
col : # | Player | To | Fee | Date
row 1 : 1 | Alan Hutton | Tottenham Hotspur | 9,000,000 | 30 January 2008
row 2 : 2 | Giovanni van Bronckhorst | Arsenal | 8,500,000 | 20 June 2001
row 3 : 3 | Jean-Alain Boumsong | Newcastle United | 8,000,000 | 1 January 2005
row 4 : 4 | Carlos Cuellar | Aston Villa | 7,800,000 | 12 August 2008
row 5 : 5 | Barry Ferguson | Blackburn Rovers | 7,500,000 | 29 August 2003
row 6 : 6 | Tore Andre Flo | Sunderland | 6,750,000 | 30 August 2002
row 7 : 7 | Trevor Steven | Marseille | 5,585,000 | 31 August 1991
row 8 : 8 | Nikica Jelavic | Everton | 5,500,000 | 31 January 2012
row 9 : 9 | Duncan Ferguson | Everton | 4,200,000 | 11 December 1994
row 10 : 10 | Claudio Reyna | Sunderland | 4,000,000 | 7 December 2001


Passages linked to row 1:
- Alan Hutton Alan Hutton ( born 30 November 1984 ) is a Scottish former professional footballer , who played as a right back . Hutton started his career with Rangers , and won the league title in 2005 . He moved to English football with Tottenham Hotspur in 2008 , and helped them win the League Cup later that year . After a loan spell with Sunderland in 2010 , Hutton joined Aston Villa in 2011 . He initally struggled with Villa , and was loaned to Mallorca , Nottingham Forest and Bolton Wanderers . Hutton then enjoyed a renaissance with Villa , and left the club after helping them win promotion in 2019 . Hutton made 50 appearances for the Scotland national team between 2007 and 2016 .
- Tottenham Hotspur F.C. Tottenham Hotspur Football Club , commonly referred to as Tottenham ( /ˈtɒtənəm/ ) or Spurs , is an English professional football club in Tottenham , London , that competes in the Premier League .
Passages linked to row 2:
- Jean-Alain Boumsong Jean-Alain Boumsong Somkong ( born 14 December 1979 ) is a former professional football defender , including French international . He is known for his physical strength , pace and reading of the game .
- Newcastle United F.C Newcastle United Football Club is an English professional football club based in Newcastle upon Tyne , Tyne and Wear , that plays in the Premier League , the top tier of English football . Founded in 1892 by the merger of Newcastle East End and Newcastle West End .
*/
Question: when was the third highest paid Rangers F.C . player born ?
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
The answer is: 14 December 1979.
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Here is the table and passages to answer this question. Answer the question with a short span. The answer needs to be just in a few words.
/*
{table_and_passages}
*/
Question: {question}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
The answer is:
"""

chain_of_thought_qa_prompt = """
<|start_header_id|>user<|end_header_id|>
Here is the table and passages to answer this question. Please provide your explanation first, then
answer the question in a short phrase starting by 'therefore, the answer is:'{sensitive_keywords_instruction}
/*
Title : List of politicians, lawyers, and civil servants educated at Jesus College, Oxford
col : Name | M | G | Degree | Notes
row 1 : Lalith Athulathmudali | 1955 | 1960 | BA Jurisprudence ( 2nd , 1958 ) , BCL ( 2nd , 1960 ) | President of the Oxford Union ( 1958 ) ; a Sri Lankan politician ; killed by the Tamil Tigers in 1993
row 2 : Neal Blewett ( HF ) | 1957 | 1959 | BA PPE ( 2nd ) | Member of the Australian House of Representatives ( 1977-1994 ) , Government Minister ( 1983-1994 ) , High Commissioner to the UK ( 1994-1998 )
row 3 : Joseph Clearihue | 1911 | 1914 | BA Jurisprudence ( 2nd , 1913 ) , BCL ( 3rd , 1914 ) | Canadian Rhodes scholar ; later became a member of the Legislative Assembly of British Columbia and a county court judge ; also chairman of the council of Victoria College , British Columbia ( which became the University of Victoria under his leadership )


Passages linked to row 1:
- Lalith Athulathmudali Lalith William Samarasekera Athulathmudali , PC ( Sinhala : ලලිත් ඇතුලත්මුදලි ; 26 November 1936 - 23 April 1993 ) , known as Lalith Athulathmudali , was Sri Lankan statesman . He was a prominent member of the United National Party , who served as Minister of Trade and Shipping ; Minister National Security and Deputy Minister of Defence ; Minister of Agriculture , Food and Cooperatives and finally Minister of Education . 
- Law degree A law degree is an academic degree conferred for studies in law . Such degrees are generally preparation for legal careers ; but while their curricula may be reviewed by legal authority , they do not themselves confer a license . A legal license is granted ( typically by examination ) and exercised locally ; while the law degree can have local , international , and world-wide aspects .
Passages linked to row 3:
- Joseph Clearihue Joseph Badenoch Clearihue ( 1887-1976 ) was a Canadian lawyer , judge , academic and politician .
- University of Victoria The University of Victoria ( UVic ) is a public comprehensive research university located in the Greater Victoria municipalities of Oak Bay and Saanich , British Columbia , Canada . The university traces its roots to Victoria College , an institution established in 1903 .
*/
Question: what is the full name of the Jesus College alumni who graduated in 1960 ?
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
Explanation: Lalith Athulathmudali graduated in 1960, and his full name is Lalith William Samarasekera Athulathmudali. Therefore, the answer is: Lalith William Samarasekera Athulathmudali.
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Here is the table and passages to answer this question. Please provide your explanation first, then
answer the question in a short phrase starting by ’therefore, the answer is:’
/*
Title : List of Rangers F.C. records and statistics
col : # | Player | To | Fee | Date
row 1 : 1 | Alan Hutton | Tottenham Hotspur | 9,000,000 | 30 January 2008
row 2 : 2 | Giovanni van Bronckhorst | Arsenal | 8,500,000 | 20 June 2001
row 3 : 3 | Jean-Alain Boumsong | Newcastle United | 8,000,000 | 1 January 2005
row 4 : 4 | Carlos Cuellar | Aston Villa | 7,800,000 | 12 August 2008
row 5 : 5 | Barry Ferguson | Blackburn Rovers | 7,500,000 | 29 August 2003
row 6 : 6 | Tore Andre Flo | Sunderland | 6,750,000 | 30 August 2002
row 7 : 7 | Trevor Steven | Marseille | 5,585,000 | 31 August 1991
row 8 : 8 | Nikica Jelavic | Everton | 5,500,000 | 31 January 2012
row 9 : 9 | Duncan Ferguson | Everton | 4,200,000 | 11 December 1994
row 10 : 10 | Claudio Reyna | Sunderland | 4,000,000 | 7 December 2001


Passages linked to row 1:
- Alan Hutton Alan Hutton ( born 30 November 1984 ) is a Scottish former professional footballer , who played as a right back . Hutton started his career with Rangers , and won the league title in 2005 . He moved to English football with Tottenham Hotspur in 2008 , and helped them win the League Cup later that year . After a loan spell with Sunderland in 2010 , Hutton joined Aston Villa in 2011 . He initally struggled with Villa , and was loaned to Mallorca , Nottingham Forest and Bolton Wanderers . Hutton then enjoyed a renaissance with Villa , and left the club after helping them win promotion in 2019 . Hutton made 50 appearances for the Scotland national team between 2007 and 2016 .
- Tottenham Hotspur F.C. Tottenham Hotspur Football Club , commonly referred to as Tottenham ( /ˈtɒtənəm/ ) or Spurs , is an English professional football club in Tottenham , London , that competes in the Premier League .
Passages linked to row 2:
- Jean-Alain Boumsong Jean-Alain Boumsong Somkong ( born 14 December 1979 ) is a former professional football defender , including French international . He is known for his physical strength , pace and reading of the game .
- Newcastle United F.C Newcastle United Football Club is an English professional football club based in Newcastle upon Tyne , Tyne and Wear , that plays in the Premier League , the top tier of English football . Founded in 1892 by the merger of Newcastle East End and Newcastle West End .
*/
Question: when was the third highest paid Rangers F.C . player born ?
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
Explanation: Jean-Alain Boumsong was the third highest paid Rangers F.C. player, and he was born on 14 December 1979. Therefore, the answer is: 14 December 1979.
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Here is the table and passages to answer this question. Please provide your explanation first, then
answer the question in a short phrase starting by ’therefore, the answer is:’
/*
{table_and_passages}
*/
Question: {question}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
Explanation: 
"""
