# Gene set enrichment analysis for the FFPE-RNA-seq data

import pandas as pd
import os
import numpy as np
import gseapy as gp
from gseapy import gseaplot
import matplotlib.pyplot as plt
import pdb

def rank_genes(df):
    resOrdered = df.sort_values("pvalue") # order results by p values
    resOrdered['fcsign'] = np.sign(resOrdered['log2FoldChange'])
    resOrdered['logP']=-np.log10(resOrdered['pvalue'])
    resOrdered['metric']= resOrdered['logP']/resOrdered['fcsign']
    return resOrdered

source_dir = "."
data_dir = os.path.join(source_dir, "sample_data")
save_dir = os.path.join(source_dir, "output")
plot_dir = save_dir

os.makedirs(save_dir, exist_ok = True)
os.makedirs(plot_dir, exist_ok = True)

df_deg = pd.read_csv(os.path.join(data_dir, "DESeq_result.csv"), index_col = 0) # DESeq2 result

rank_deg = rank_genes(df_deg)
rank_deg.head(50)
rank_deg = rank_deg[['metric']]


gene_sets = ['GO_Biological_Process_2021', 'KEGG_2021_Human']
pre_res = gp.prerank(rnk=rank_deg, # or rnk = rnk,
                    gene_sets=gene_sets,
                    threads=4,
                    min_size=5,
                    max_size=1000,
                    permutation_num=1000, # reduce number to speed up testing
                    outdir=None, # don't write to disk
                    seed=6,
                    verbose=True, # see what's going on behind the scenes
                    )
df_gsea = pre_res.res2d
print(df_gsea.shape)
df_gsea.to_csv(os.path.join(save_dir, "prerank_gsea_GO_KEGG.csv"), index = False)

# Read out GSEA results
df_sig = df_gsea[(df_gsea['NOM p-val'] < 0.05) & (df_gsea['FDR q-val'] < 0.2)]

df_up = df_sig[df_sig['NES'] > 0]
df_down = df_sig[df_sig['NES'] < 0]
df_up = df_up.sort_values("NES", ascending = False)
df_down = df_down.sort_values("NES", ascending = True)
df_sig = pd.concat([df_up, df_down])
df_sig['Category'] = df_sig['Term'].apply(lambda x: x.split("__")[0])
df_sig['Term'] = df_sig['Term'].apply(lambda x: x.split("__")[1])
df_sig['Tag %'] = df_sig['Tag %'].astype(str)
df_sig = df_sig[['Category', 'Term', 'NES', 'NOM p-val', 'FDR q-val', "Gene %", "Lead_genes"]]
df_sig.to_csv(os.path.join(save_dir, "significant_gsea_GO_KEGG.csv"), index = False)

terms = df_gsea['Term']
selected_terms = ['GO_Biological_Process_2021__positive regulation of T cell activation (GO:0050870)',
                  'GO_Biological_Process_2021__antigen receptor-mediated signaling pathway (GO:0050851)',
                  'GO_Biological_Process_2021__regulation of B cell activation (GO:0050864)', 
                  'GO_Biological_Process_2021__response to interferon-gamma (GO:0034341)',
                  'GO_Biological_Process_2021__regulation of interleukin-6 production (GO:0032675)',
                  'GO_Biological_Process_2021__regulation of cytokine production (GO:0001817)',
                  'GO_Biological_Process_2021__regulation of interleukin-2 production (GO:0032663)',
                  'GO_Biological_Process_2021__dendritic cell differentiation (GO:0097028)',
                  'GO_Biological_Process_2021__positive regulation of T cell migration (GO:2000406)',
                  'GO_Biological_Process_2021__immune response-activating cell surface receptor signaling pathway (GO:0002429)',
                  'GO_Biological_Process_2021__toll-like receptor signaling pathway (GO:0002224)',
                  'GO_Biological_Process_2021__unsaturated fatty acid metabolic process (GO:0033559)',
                  'GO_Biological_Process_2021__tight junction assembly (GO:0120192)',
                  'GO_Biological_Process_2021__cell-cell adhesion mediated by cadherin (GO:0044331)',
                  'KEGG_2021_Human__Th1 and Th2 cell differentiation',
                  'KEGG_2021_Human__Antigen processing and presentation',
                  'KEGG_2021_Human__Natural killer cell mediated cytotoxicity',
                  'KEGG_2021_Human__Cytokine-cytokine receptor interaction',
                  "KEGG_2021_Human__PD-L1 expression and PD-1 checkpoint pathway in cancer",
                  "KEGG_2021_Human__T cell receptor signaling pathway",
                  "KEGG_2021_Human__NF-kappa B signaling pathway",]

overlap_terms = list(set(df_gsea['Term']) & set(selected_terms))
print(len(overlap_terms))

df_select = df_gsea[df_gsea['Term'].isin(selected_terms)]
df_select.to_csv(os.path.join(save_dir, "selected_gsea_GO_KEGG.csv"), index = False)

for term in selected_terms:
    plt.figure()
    new_term = term.split("__")[1]
    pre_res.results[new_term] = pre_res.results[term].copy()
    axs = gseaplot(rank_metric=pre_res.ranking, term=new_term, **pre_res.results[new_term])
    plt.savefig(os.path.join(plot_dir, f"{term}.png"), bbox_inches = "tight")
    plt.close()


