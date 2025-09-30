#数据集下载
from modelscope.msdatasets import MsDataset
ds =  MsDataset.load('ddzhu123/seq-monkey')
#您可按需配置 subset_name、split，参照“快速使用”示例代码
print(ds['train'][0])
