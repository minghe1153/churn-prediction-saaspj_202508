# 必要なライブラリが読み込まれているか確認
library(dagitty)
library(ggdag)
library(ggplot2)

# DAG定義（コメントを削除）
simple_dag <- dagitty('
dag {
  Geography [exposure,pos="0,0"]
  AgeOver40 [pos="1,0"]
  Income [pos="2,0"]
  
  BalanceRisk [pos="0,1"]
  EngagementLevel [pos="1,1"]
  
  ChurnIntention [latent,pos="0,2"]
  Exited [outcome,pos="1,2"]
  
  Geography -> BalanceRisk
  Geography -> Exited
  
  AgeOver40 -> BalanceRisk
  AgeOver40 -> ChurnIntention
  
  Income -> BalanceRisk
  Income -> EngagementLevel
  
  BalanceRisk -> ChurnIntention
  EngagementLevel -> ChurnIntention
  
  ChurnIntention -> Exited
}
')

# 可視化
p2 <- ggdag(simple_dag, layout = "sugiyama") + 
  theme_dag() +
  geom_dag_edges(edge_color = "darkgreen", 
                 edge_width = 0.5) +  
  geom_dag_node(aes(color = name == "Exited"), size = 15, show.legend = FALSE) +
  scale_color_manual(values = c("lightcyan", "tomato")) +
  geom_dag_text(color = "black", size = 4) +  
  ggtitle("DAG") +  
  theme(
    plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
    plot.margin = unit(c(1, 1, 1, 1), "cm")
  ) +
  coord_cartesian(clip = "off") +
  expand_limits(y = c(-0.5, 2.5))

print(p2)

