# Differential gene expression analysis using DESeq2

library(DESeq2)
library(ggplot2)
library(ggrepel) 

source_dir = getwd()
data_dir = file.path(source_dir, "sample_data")
save_dir = file.path(source_dir, "output")
dir.create(save_dir, showWarnings = FALSE)

# count_matrix
df_count = read.csv(file.path(data_dir, "Count.csv"), row.names = 1) # (19789, 129)

# Sample annotation matrix
metadata = read.csv(file.path(data_dir, "metadata.csv"), row.names = 1)

# PCA plot
coldata = metadata
table(coldata$condition)

cts = df_count[, rownames(coldata)]
dim(cts) # 19789, 33
cts = round(as.matrix(cts))

all(rownames(coldata) %in% colnames(cts))
all(rownames(coldata) == colnames(cts))

dds <- DESeqDataSetFromMatrix(countData = cts,
                              colData = coldata,
                              design = ~ condition)

dds$condition <- relevel(dds$condition, ref = "non_res") # set reference group to non-responders

# Filter genes with low counts
smallestGroupSize <- 1
keep <- rowSums(counts(dds) >= 10) >= smallestGroupSize
dds <- dds[keep,] 
print(dim(dds)) # dim: [1] 18026  35

vsd <- vst(dds, blind=FALSE)
head(assay(vsd), 3)
pdf(file.path(save_dir, paste0("pca_all.pdf")), width = 6, height = 6)
plotPCA(vsd, intgroup=c("condition")) +
  theme_minimal() +  # Use a minimal theme +
  geom_text_repel(aes(label = name),
                  max.overlaps = 50000,                    # 最大覆盖率，当点很多时，有些标记会被覆盖，调大该值则不被覆盖，反之。
                  size=2,                                  # 字体大小
                  box.padding=unit(0.5,'lines'),           # 标记的边距
                  point.padding=unit(0.1, 'lines'),
                  segment.color='black',                   # 标记线条的颜色
                  show.legend=FALSE)
dev.off()

dds <- DESeq(dds)
res <- results(dds)
resultsNames(dds)
resLFC <- lfcShrink(dds, coef="condition_res_vs_non_res", type="apeglm")
resLFC
resOrdered <- resLFC[order(resLFC$pvalue),] # order results by p values
resOrdered = as.data.frame(resOrdered)
write.csv(resOrdered, file.path(save_dir, "DESeq_result.csv"))