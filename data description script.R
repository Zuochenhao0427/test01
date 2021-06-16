library(tidyverse)
library(plotly)
library(stringr)
library(lubridate)

df=read_csv('数据已合并.csv') # 原始数据73616个数据点

## 单个代码对应单支基金？N, 独立基金(拥有独立代码的)一共有623个，554个简称中有69个简称对应2个代码。
df %>% select(`代码`,`简称`) %>% unique() %>% nrow() # 623
df %>% select(`代码`) %>% unique() %>% nrow() # 623
df %>% select(`简称`) %>% unique() %>% nrow() # 554
df %>% select(`代码`,`简称`) %>% unique() %>% ungroup() %>%
  group_by(`简称`) %>% count() %>% filter(n!=1) %>% nrow() # 69
#######################################
########### Before NA remove ##########
#######################################
## 一共有几支基金？按单独代码计算，一共有623支基金。
## 单支基金对应多少个数据点(未dropna以前)？有334支基金数据点少于50个，其中304支基金数据点不足10个。
df %>% group_by(`代码`) %>% count() -> plt1
ggplot(plt1) +
  geom_histogram(aes(`n`), binwidth = 10) -> p1
ggplotly(p1) 

## drop na '--' ---> 以单位净值为sample
df$`周单位净值(元)` = ifelse(df$`周单位净值(元)`=="--", NA, df$`周单位净值(元)`)
df %>% select(c(1:3), 6) %>% na.omit() -> df1 # 去掉NA一共68000+数据点

#######################################
########### After NA remove ###########
#######################################
## 一共有几支基金？一共328支基金。
df1 %>% select(`代码`) %>% unique() %>% nrow() # 328
## 单支基金对应多少个数据点？有85%的基金单支拥有50 ~ 400个数据点, 其中有25.8%的基金单支拥有200~250个数据点。
df1 %>% group_by(`代码`) %>% count() -> plt2
ggplot(plt2) +
  geom_histogram(aes(`n`), binwidth = 50) -> p2
ggplotly(p2) 
## 每年有多少个数据点？由04年至18年，数据点数量由最初的1个至15674个逐年增多。
df1 %>% mutate(年份=year(时间)) -> df_year
df_year %>% group_by(年份) %>% count() -> plt3
ggplot(plt3) +
  geom_line(aes(x=`年份`, y=`n`)) -> p3
ggplotly(p3) 
## 每年统计多少支基金？由04年至18年，基金数量由最初的1支至328支逐年增多，13年基金数量超过100支，14年后基金数量猛增。
df_year %>% select(年份, 代码) %>% unique() %>% group_by(年份) %>% count() -> plt4
ggplot(plt4) +
  geom_line(aes(x=`年份`, y=`n`)) -> p4
ggplotly(p4)

# 导出数据
write_csv(df1, 'funds_data.csv')
