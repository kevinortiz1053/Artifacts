# NOTE: To load data, you must download both the extract's data and the DDI
# and also set the working directory to the folder with these files (or change the path below).

if (!require("ipumsr")) stop("Reading IPUMS data into R requires the ipumsr package. It can be installed using the following command: install.packages('ipumsr')")

library(ipumsr)
ddi <- read_ipums_ddi("usa_00010.xml")
data <- read_ipums_micro(ddi)
data = subset(data, select = c(STATEFIP, METRO, CITY, OWNERSHP, MORTGAGE, FOODSTMP, FAMSIZE, NCHILD, NSIBS, SEX, AGE, MARST, RACE, HISPAN, BPL, CITIZEN, YRIMMIG, YRSUSA1, LANGUAGE, SPEAKENG, EDUC, GRADEATT, SCHLTYPE, DEGFIELD, EMPSTAT, OCC, IND, UHRSWORK, INCTOT, VETSTAT, PWSTATE2, TRANWORK, CARPOOL))
filtered_data = data[data$INCTOT < 9999998 & data$INCTOT > 1, ]
filtered_data = filtered_data[filtered_data$YRIMMIG > 1949, ]
filtered_data = subset(filtered_data, select = c(STATEFIP, COUNTYFIP, METRO, CITY, CITYPOP, OWNERSHP, MORTGAGE, FOODSTMP, FAMSIZE, NCHILD, NSIBS, SEX, AGE, MARST, RACE, HISPAN, BPL, CITIZEN, YRSUSA1, LANGUAGE, SPEAKENG, EDUC, GRADEATT, SCHLTYPE, DEGFIELD, EMPSTAT, OCC, IND, UHRSWORK, INCTOT, VETSTAT, PWSTATE2, TRANWORK, CARPOOL))
filtered_data = filtered_data[filtered_data$BPL == 110 | filtered_data$BPL == 200 | filtered_data$BPL == 210 | filtered_data$BPL == 250 | filtered_data$BPL == 300, ]

income = subset(data, select = c(INCTOT))
income = income[with(income, order(INCTOT, decreasing = TRUE)), ]
immi = subset(data, select = c(YRIMMIG))
immi = immi[with(immi, order(YRIMMIG, decreasing = TRUE)), ]
birth = subset(data, select = c(BPL))
birth = birth[with(birth, order(BPL, decreasing = TRUE)), ]
birth = birth[birth$BPL == 110 | birth$BPL == 200 | birth$BPL == 210 | birth$BPL == 250 | birth$BPL == 300, ]
View(wild)
wild = subset(filtered_data, select = c(COUNTYFIP))


intddi <- read_ipums_ddi("DDI_codebook_extract_04.xml")
intdata <- read_ipums_micro(intddi)

#########################################################################
columns_data = colnames(data)
col_names_data = data.frame(as.list(columns_data))
write_xlsx(col_names_data,"/Users/charles/Desktop/Data Science/Census Exploration/Data/column_names.xlsx")
length_data = ncol(data)
print(length_data)

x = data[data$INCTOT == 9999999, ]
y = data[data$INCTOT < 1, ]
z = data[data$INCTOT < 9999998 & data$INCTOT > 0, ]

a = data[data$INCWAGE == 999998, ]
b = data[data$INCWAGE == 0, ]
c = data[data$INCWAGE < 999999 & data$INCWAGE > 0, ]


print(nrow(b))
View(col_names_data)


filtered_inctot_data = data[data$INCTOT <  ]
filtered_incwage_data = data[data$INCWAGE]

#########################################################################


#basic = subset(data, select = c(YEAR, MARST, LANGUAGED, CITIZEN, AGE, INCTOT, OCC, BPL, YRIMMIG, HISPAND))
#basic1 = basic[basic$INCTOT < 9999999 & basic$YRIMMIG > 0, ]

#success_criteria = subset(filtered_data, select = c(MULTYEAR, SERIAL, STATEFIP, METRO, MET2013, CITY, CITYPOP, OWNERSHPD, MORTGAGE, RENT, RENTGRS, COSTELEC, COSTGAS, COSTWATR, COSTFUEL, HHINCOME, NCHILD, NCHLT5, SEX, AGE, MARST, MARRNO, RACE, RACED, HISPAN, HISPAND, BPL, BPLD, CITIZEN, YRIMMIG, YRSUSA1, LANGUAGE, LANGUAGED, SPEAKENG, SCHOOL, EDUC, EDUCD, DEGFIELD, EMPSTAT, OCC, IND, WKSWORK2, UHRSWORK, INCTOT, FTOTINC, POVERTY, OCCSCORE, PWSTATE2, TRANWORK, CARPOOL))
#success_criteria = subset(filtered_data, select = c(MULTYEAR, STATEFIP, METRO, OWNERSHPD, MORTGAGE, NCHILD, SEX, AGE, MARST, RACE, HISPAN, BPL, CITIZEN, YRIMMIG, YRSUSA1, LANGUAGE, SPEAKENG, SCHOOL, EDUC, EMPSTAT, OCC, IND, UHRSWORK, INCTOT, POVERTY, TRANWORK, CARPOOL))
success_criteria = subset(filtered_data, select = c(INCTOT, STATEFIP, METRO, OWNERSHPD, MORTGAGE, NCHILD, SEX, AGE, MARST, RACE, HISPAN, BPL, CITIZEN, YRSUSA1, LANGUAGE, SPEAKENG, SCHOOL, EDUC, EMPSTAT, OCC, IND, UHRSWORK, POVERTY, TRANWORK, CARPOOL))
success_criteria = subset(filtered_data, select = c(INCTOT, STATEFIP, MARST, CITIZEN, SPEAKENG, EDUC, OCC, CARPOOL))
tier_4_data = success_criteria[success_criteria$INCTOT >= 90000, ]
tier_4_data = subset(tier_4_data, select = c(STATEFIP, METRO, OWNERSHPD, MORTGAGE, NCHILD, SEX, AGE, MARST, RACE, HISPAN, BPL, CITIZEN, YRSUSA1, LANGUAGE, SPEAKENG, SCHOOL, EDUC, EMPSTAT, OCC, IND, UHRSWORK, POVERTY, TRANWORK, CARPOOL))

tier_3_data = success_criteria[success_criteria$INCTOT >= 60000 & success_criteria$INCTOT < 90000, ]
tier_3_data = subset(tier_3_data, select = c(STATEFIP, METRO, OWNERSHPD, MORTGAGE, NCHILD, SEX, AGE, MARST, RACE, HISPAN, BPL, CITIZEN, YRSUSA1, LANGUAGE, SPEAKENG, SCHOOL, EDUC, EMPSTAT, OCC, IND, UHRSWORK, POVERTY, TRANWORK, CARPOOL))

tier_2_data = success_criteria[success_criteria$INCTOT >= 30000 & success_criteria$INCTOT < 60000, ]
tier_2_data = subset(tier_2_data, select = c(STATEFIP, METRO, OWNERSHPD, MORTGAGE, NCHILD, SEX, AGE, MARST, RACE, HISPAN, BPL, CITIZEN, YRSUSA1, LANGUAGE, SPEAKENG, SCHOOL, EDUC, EMPSTAT, OCC, IND, UHRSWORK, POVERTY, TRANWORK, CARPOOL))

tier_1_data = success_criteria[success_criteria$INCTOT > -9999999 & success_criteria$INCTOT < 30000, ]
tier_1_data = subset(tier_1_data, select = c(STATEFIP, METRO, OWNERSHPD, MORTGAGE, NCHILD, SEX, AGE, MARST, RACE, HISPAN, BPL, CITIZEN, YRSUSA1, LANGUAGE, SPEAKENG, SCHOOL, EDUC, EMPSTAT, OCC, IND, UHRSWORK, POVERTY, TRANWORK, CARPOOL))


tier_4_data1 = tier_4_data[sample(nrow(tier_4_data), size = 5, replace = FALSE), ]
tier_3_data1 = tier_3_data[sample(nrow(tier_3_data), size = 5, replace = FALSE), ]
tier_2_data1 = tier_2_data[sample(nrow(tier_2_data), size = 800000, replace = FALSE), ]
tier_1_data1 = tier_1_data[sample(nrow(tier_1_data), size = 5, replace = FALSE), ]


View(tier_3_data1)


temp = subset(filtered_data, select = c(INCTOT))
temp = temp[!is.na(temp$INCTOT), ]
wild = success_criteria[with(success_criteria, order(MULTYEAR, decreasing = TRUE)), ]
View(census_data_2021)


census_data_2021_v2 = success_criteria[success_criteria$MULTYEAR == 2021, ]
census_data_2020_v2 = success_criteria[success_criteria$MULTYEAR == 2020, ]
census_data_2019_v2 = success_criteria[success_criteria$MULTYEAR == 2019, ]
census_data_2018_v2 = success_criteria[success_criteria$MULTYEAR == 2018, ]
census_data_2017_v2 = success_criteria[success_criteria$MULTYEAR == 2017, ]
census_data_2016_v2 = success_criteria[success_criteria$MULTYEAR == 2016, ]
census_data_2015_v2 = success_criteria[success_criteria$MULTYEAR == 2015, ]
census_data_2014_v2 = success_criteria[success_criteria$MULTYEAR == 2014, ]
census_data_2013_v2 = success_criteria[success_criteria$MULTYEAR == 2013, ]
census_data_2012_v2 = success_criteria[success_criteria$MULTYEAR == 2012, ]
census_data_2011_v2 = success_criteria[success_criteria$MULTYEAR == 2011, ]
census_data_2010_v2 = success_criteria[success_criteria$MULTYEAR == 2010, ]
census_data_2009_v2 = success_criteria[success_criteria$MULTYEAR == 2009, ]
census_data_2008_v2 = success_criteria[success_criteria$MULTYEAR == 2008, ]
census_data_2007_v2 = success_criteria[success_criteria$MULTYEAR == 2007, ]
census_data_2006_v2 = success_criteria[success_criteria$MULTYEAR == 2006, ]
census_data_2005_v2 = success_criteria[success_criteria$MULTYEAR == 2005, ]

rows = floor(nrow(success_criteria) / 10)
succ_criteria_one = success_criteria[0:rows, ]
index = 2 * rows
index1 = 3 * rows
index2 = 4 * rows
index3 = 5 * rows
index4 = 6 * rows
index5 = 7 * rows
index6 = 8 * rows
index7 = 9 * rows
index8 = 10 * rows - 1
succ_criteria_two = success_criteria[rows: index, ]
succ_criteria_three = success_criteria[index: index1, ]
succ_criteria_four = success_criteria[index1: index2, ]
succ_criteria_five = success_criteria[index2: index3, ]
succ_criteria_six = success_criteria[index3: index4, ]
succ_criteria_seven = success_criteria[index4: index5, ]
succ_criteria_eight = success_criteria[index5: index6, ]
succ_criteria_nine = success_criteria[index6: index7, ]
succ_criteria_ten = success_criteria[index7: index8, ]



########### FINDING QUARTILES ################
View(wild)

income = subset(wild, select = c(INCTOT))
income_list = income$INCTOT

median_income = median(income_list) # median: 21,600
median_index = which(income_list == 21600) # median index: 1,327,488
median_index[[3492]] # last index having the same median value is 1,330,978

top_half = income[0:1327488, ]
thirdq_income = median(top_half) # third quartile: 50,000
thirdq_index = which(income_list == 50000) # third quartile index 623,718

lower_half = income[1327488:2656204, ]
firstq_income = median(lower_half) # first quartile income: 6000
firstq_index = which(income_list == 6000) # first quartile index 1,991,669

# fourth quartile: 1,629,000


install.packages("writexl")
library("writexl")
write_xlsx(tier_2_data1,"/Users/charles/Desktop/Data Science/Personal Projects/Census Exploration/Data/census_data.csv")
write.csv(filtered_data, "/Users/charles/Desktop/Data Science/Personal Projects/Census Exploration/Data/census_data.csv")


