detect_aggregation_query_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>Using f_agg() API, return True to detect when a natural language query involves performing aggregation operations (max, min, avg, group by). 
Strictly follow the format of the below examples. Please provide your explanation first, then answer the question in a short phrase starting by ’Therefore, the answer is:’

question: when was the third highest paid Rangers F.C . player born ?
Explanation: The question involves finding the birth date of the third highest paid player, which requires aggregation to find the third highest paid player. Therefore, the answer is : f_agg([True])

question: what is the full name of the Jesus College alumni who graduated in 1960 ?
Explanation: The question involves finding the full name of the alumni who graduated in 1960, which does not require aggregation. Therefore, the answer is : f_agg([False])

question: how tall , in feet , is the Basketball personality that was chosen as MVP most recently ?
Explanation:  The question involves finding the most recent MVP winner, which requires aggregation to identify the relevant player. Therefore, the answer is : f_agg([True])

question: what is the highest best score series 7 of Ballando con le Stelle for the best dancer born 3 July 1969 ?
Explanation: The question involves finding the highest score in a series for a specific dancer, which requires aggregation. Therefore, the answer is : f_agg([True])

question: which conquerors established the historical site in England that attracted 2,389,548 2009 tourists ?
Explanation: The question involves identifying the conquerors who established a historical site, which does not require aggregation. Therefore, the answer is : f_agg([False])

question: what is the NYPD Blue character of the actor who was born on January 29 , 1962 ?
Explanation: The question involves finding the character played by an actor born on a specific date, which does not require aggregation. Therefore, the answer is : f_agg([False])
<|eot_id|><|start_header_id|>user<|end_header_id|>
question: {question}
Explanation: <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

select_row_wise_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>Using f_row() API to select relevant rows in the given table and linked passages that support or oppose the question.
Strictly follow the format of the below example. Please provide your explanation first, then select relevant rows in a short phrase starting by ’Therefore, the relevant rows are:’

/*
table caption : list of rangers f.c. records and statistics
col : # | player | to | fee | date
row 1 : 1 | alan hutton | tottenham hotspur | 9,000,000 | 30 january 2008
row 2 : 2 | giovanni van bronckhorst | arsenal | 8,500,000 | 20 june 2001
row 3 : 3 | jean-alain boumsong | newcastle united | 8,000,000 | 1 january 2005
row 4 : 4 | carlos cuellar | aston villa | 7,800,000 | 12 august 2008
row 5 : 5 | barry ferguson | blackburn rovers | 7,500,000 | 29 august 2003
*/

/*
passages linked to row 1
- alan hutton: alan hutton ( born 30 november 1984 ) is a scottish former professional footballer , who played as a right back . hutton started his career with rangers , and won the league title in 2005 . he moved to english football with tottenham hotspur in 2008 , and helped them win the league cup later that year .
- tottenham hotspur f.c.: tottenham hotspur football club , commonly referred to as tottenham ( /ˈtɒtənəm/ ) or spurs , is an english professional football club in tottenham , london , that competes in the premier league .
passages linked to row 2
- giovanni van bronckhorst: giovanni christiaan van bronckhorst oon ( dutch pronunciation : [ ɟijoːˈvɑni vɑm ˈbrɔŋkɦɔrst ] ( listen ) ; born 5 february 1975 ) , also known by his nickname gio , is a retired dutch footballer and currently the manager of guangzhou r & f . formerly a midfielder , he moved to left back later in his career .
- arsenal f.c.: arsenal football club is a professional football club based in islington , london , england , that plays in the premier league , the top flight of english football . the club has won 13 league titles , a record 13 fa cups , 2 league cups , 15 fa community shields , 1 league centenary trophy , 1 uefa cup winners ' cup and 1 inter-cities fairs cup . 
passages linked to row 3
- jean-alain boumsong: jean-alain boumsong somkong ( born 14 december 1979 ) is a former professional football defender , including french international . he is known for his physical strength , pace and reading of the game .
- newcastle united f.c.: newcastle united football club is an english professional football club based in newcastle upon tyne , tyne and wear , that plays in the premier league , the top tier of english football . founded in 1892 by the merger of newcastle east end and newcastle west end . 
passages linked to row 4
- carlos cuéllar: carlos javier cuéllar jiménez ( spanish pronunciation : [ ˈkaɾlos ˈkweʎaɾ ] ; born 23 august 1981 ) is a spanish professional footballer who plays for israeli club bnei yehuda . mainly a central defender , he can also operate as a right back .
- aston villa: aston villa football club ( nicknamed villa ) is an english professional football club based in aston , birmingham . the club competes in the premier league , the top tier of the english football league system . founded in 1874 , they have played at their home ground , villa park , since 1897 .
*/

question : when was the third highest paid rangers f.c . player born ?
Explanation : The third-highest paid Rangers F.C. player, Jean-Alain Boumsong (row 3). Therefore, the relevant rows are : f_row([row 3])

/*
table caption : missouri valley conference men's basketball tournament
col : year | mvc champion | score | runner-up | tournament mvp | venue ( and city )
row 1 : 1994 | southern illinois | 77-74 | northern iowa | cam johnson , northern iowa | st. louis arena ( st. louis , missouri )
row 2 : 1996 | tulsa | 60-46 | bradley | shea seals , tulsa | kiel center ( st. louis , missouri )
*/

/*
passages linked to row 1
- southern illinois salukis men's basketball: the southern illinois salukis men 's basketball team represents southern illinois university carbondale in carbondale , illinois . the salukis compete in the missouri valley conference , and they play their home games at banterra center . as of march 2019 , saluki hall of fame basketball player , bryan mullins , has become the newest head coach of the southern illinois basketball program .
- northern iowa panthers men's basketball: the northern iowa panthers men 's basketball team represents the university of northern iowa , located in cedar falls , iowa , in ncaa division i basketball competition . uni is currently a member of the missouri valley conference .
passages linked to row 2
- tulsa golden hurricane men's basketball: the tulsa golden hurricane men 's basketball team represents the university of tulsa in tulsa , in the u.s. state of oklahoma . the team participates in the american athletic conference . the golden hurricane hired frank haith from missouri on april 17 , 2014 to replace danny manning , who had resigned to take the wake forest job after the 2013-14 season . the team has long been successful , especially since the hiring of nolan richardson in 1980 . 
- bradley braves men's basketball: the bradley braves men 's basketball team represents bradley university , located in peoria , illinois , in ncaa division i basketball competition . they compete as a member of the missouri valley conference . the braves are currently coached by brian wardle and play their home games at carver arena . bradley has appeared in nine ncaa tournaments , including two final fours , finishing as the national runner-up in 1950 and 1954 . 
*/

question : how tall , in feet , is the basketball personality that was chosen as mvp most recently ?
Explanation : The most recent MVP mentioned in the table is Shea Seals from Tulsa (row 2). Therefore, the relevant rows are : f_row([row 2])

/*
table caption : list of longest - serving soap opera actors.
col : dance | best dancer | best score | worst dancer | worst score
row 1 : boogie woogie | kaspar capparoni | 44 | barbara capponi | 27
row 2 : merengue | gedeon burkhard | 36 | paolo rossi | 25
row 3 : quickstep | kaspar capparoni | 44 | alessandro di pietro | 9
row 4 : samba | gedeon burkhard | 39 | giuseppe povia | 20
row 5 : tango | sara santostasi | 40 | gedeon burkhard | 27
*/

/*
passages linked to row 1
Title: kaspar capparoni. Content: gaspare kaspar capparoni ( born 1 august 1964 ) is an italian actor .
Title: barbara capponi. Content: the seventh series of ballando con le stelle was broadcast from 26 february to 30 april 2011 on rai 1 and was presented by milly carlucci with paolo belli and his 'big band ' .
passages linked to row 2
Title: gedeon burkhard. Content: gedeon burkhard ( born 3 july 1969 ) is a german film and television actor . although he has appeared in numerous films and tv series in both europe and the us , he is probably best recognised for his role as alexander brandtner in the austrian/german television series kommissar rex ( 1997-2001 ) , which has been aired on television in numerous countries around the world , or as corporal wilhelm wicki in the 2009 film inglourious basterds . 
Title: paolo rossi. Content: paolo rossi ( italian pronunciation : [ ˈpaːolo ˈrossi ] ; born 23 september 1956 ) is an italian former professional footballer , who played as a forward . in 1982 , he led italy to the 1982 fifa world cup title , scoring six goals to win the golden boot as top goalscorer , and the golden ball for the player of the tournament . rossi is one of only three players to have won all three awards at a world cup , along with garrincha in 1962 , and mario kempes in 1978 .
passages linked to row 3
Title: kaspar capparoni. Content: gaspare kaspar capparoni ( born 1 august 1964 ) is an italian actor .
Title: alessandro di pietr. Content: the seventh series of ballando con le stelle was broadcast from 26 february to 30 april 2011 on rai 1 and was presented by milly carlucci with paolo belli and his 'big band ' .
passages linked to row 4
Title: gedeon burkhard. Content: gedeon burkhard ( born 3 july 1969 ) is a german film and television actor . although he has appeared in numerous films and tv series in both europe and the us , he is probably best recognised for his role as alexander brandtner in the austrian/german television series kommissar rex ( 1997-2001 ) , which has been aired on television in numerous countries around the world , or as corporal wilhelm wicki in the 2009 film inglourious basterds . 
Title: giuseppe povia. Content: giuseppe povia ( italian pronunciation : [ poˈviːa ], born november 19 , 1972 ) , better known just as povia [ ˈpɔːvja ] , is an italian rock singer-songwriter .
passages linked to row 5
Title: sara santostasi. Content: sara santostasi ( born 24 january 1993 ) is an italian actress singer and dancer . she was one of the contestants in seventh series of ballando con le stelle , the italian version of dancing with the stars .
Title: gedeon burkhard. Content: gedeon burkhard ( born 3 july 1969 ) is a german film and television actor . although he has appeared in numerous films and tv series in both europe and the us , he is probably best recognised for his role as alexander brandtner in the austrian/german television series kommissar rex ( 1997-2001 ) , which has been aired on television in numerous countries around the world , or as corporal wilhelm wicki in the 2009 film inglourious basterds . 
*/

question : what is the highest best score series 7 of ballando con le stelle for the best dancer born 3 july 1969 ?
Explanation : The best dancer born on 3 July 1969 is Gedeon Burkhard (row 2, row 4). His highest best score is 39 for Samba (row 4). Therefore, the relevant rows are : f_row([row 4])
<|eot_id|><|start_header_id|>user<|end_header_id|>

/*
{table}
*/

/*
{linked_passages}
*/

question : {question}
Explanation : <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

select_passages_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>Using the f_passage() API, you can return a list of linked passages that contain partial information relevant to the question.
Strictly follow the format of the below example. Please provide your explanation first, then return a list of passages in a short phrase starting by ’Therefore, relevant passages are:’


/*
table caption : List of politicians, lawyers, and civil servants educated at Jesus College, Oxford
col : Name | M | G | Degree | Notes
row 1 : Lalith Athulathmudali | 1955 | 1960 | BA Jurisprudence ( 2nd , 1958 ) , BCL ( 2nd , 1960 ) | President of the Oxford Union ( 1958 ) ; a Sri Lankan politician ; killed by the Tamil Tigers in 1993
*/

/*
List of linked passages: ["Law degree", "Oxford Union", "Lalith Athulathmudali"]


Title: Law degree. Content: A law degree is an academic degree conferred for studies in law . Such degrees are generally preparation for legal careers ; but while their curricula may be reviewed by legal authority , they do not themselves confer a license . A legal license is granted ( typically by examination ) and exercised locally ; while the law degree can have local , international , and world-wide aspects .


Title: Oxford Union. Content: The Oxford Union Society , commonly referred to simply as the Oxford Union , is a debating society in the city of Oxford , England , whose membership is drawn primarily from the University of Oxford . Founded in 1823 , it is one of Britain 's oldest university unions and one of the world 's most prestigious private students ' societies . The Oxford Union exists independently from the university and is separate from the Oxford University Student Union .


Title: Lalith Athulathmudali. Content: Lalith William Samarasekera Athulathmudali , PC ( Sinhala : ලලිත් ඇතුලත්මුදලි ; 26 November 1936 - 23 April 1993 ) , known as Lalith Athulathmudali , was Sri Lankan statesman . He was a prominent member of the United National Party , who served as Minister of Trade and Shipping ; Minister National Security and Deputy Minister of Defence ; Minister of Agriculture , Food and Cooperatives and finally Minister of Education . 
*/

question: What is the full name of the Jesus College alumni who graduated in 1960 ?
Explanation: First, Lalith Athulathmudali graduated in 1960. Second, the linked passage titled "Lalith Athulathmudali" confirms his full name. Therefore, relevant passages are: f_passage(["Lalith Athulathmudali"])


/*
table caption : list of rangers f.c. records and statistics
col : # | player | to | fee | date
row 1 : 3 | jean-alain boumsong | newcastle united | \u00a38,000,000 | 1 january 2005
*/

/*
List of linked passages: ["Newcastle United F.C.", "Jean-Alain Boumsong"]


Title: Newcastle United F.C.. Content: Newcastle United Football Club is an English professional football club based in Newcastle upon Tyne , Tyne and Wear , that plays in the Premier League , the top tier of English football . Founded in 1892 by the merger of Newcastle East End and Newcastle West End . 


Title: Jean-Alain Boumsong. Content: Jean-Alain Boumsong Somkong ( born 14 December 1979 ) is a former professional football defender , including French international . He is known for his physical strength , pace and reading of the game .
*/

question: When was the third highest paid Rangers F.C . player born ?
Explanation: First, Jean-Alain Boumsong is listed as the third highest paid player. Second, the linked passage titled "Jean-Alain Boumsong" confirms his birthdate. Therefore, relevant passages are: f_passage(["Jean-Alain Boumsong"])


/*
table caption : tourism in england
col : national rank | site | location | visitor count ( 2009 )
row 1 : 1 | Tower of London | London | 2,389,548
*/

/*
List of linked passages: ["London", "Tower of London"]


Title: London. Content: London is the capital and largest city of England and of the United Kingdom. It has been a major settlement for two millennia and was founded by the Romans. Greater London is governed by the Mayor of London and the London Assembly. London exerts a significant impact on various sectors, including tourism, ranking among the most visited cities globally.


Title: Tower of London. Content: The Tower of London, officially Her Majesty's Royal Palace and Fortress of the Tower of London, is a historic castle located on the north bank of the River Thames in central London. It was founded towards the end of 1066 as part of the Norman Conquest of England. The White Tower, built by William the Conqueror in 1078, was a symbol of oppression inflicted upon London by the new ruling elite. The castle was used as a prison and served as a royal residence during its history.
*/

question: Which conqueror established the historical site in England that attracted 2,389,548 tourists in 2009?
Explanation: First, the Tower of London attracted 2,389,548 tourists in 2009. Second, the linked passage titled "Tower of London" confirms that it was established by Norman. Therefore, relevant passages are: f_passage(["Tower of London"])


<|eot_id|><|start_header_id|>user<|end_header_id|>

/*
{table_segment}
*/

/*
{linked_passages}
*/

**Important** Please ensure that you select all passages that are even slightly related to the question, regardless of whether they contain similar or duplicate information. Include all relevant passages without omission.

question: {question}
Explanation: <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""