============================================================
Global checkpoint (Fed → downstream)
============================================================
✅ Found: results/global_checkpoint/global_model.pt
{
  "model_name": "Qwen/Qwen2.5-0.5B",
  "num_labels": 4,
  "use_lora": true,
  "architecture": "decoder",
  "experiment_name": "vgae_augmp",
  "lora_r": 8,
  "lora_alpha": 16,
  "lora_dropout": 0.1,
  "lora_target_modules": null
}
Running: /usr/bin/python3 run_downstream_generation.py --checkpoint results/global_checkpoint --probes "data/AG News Datasets/ag_news_business_30.json" --output results/colab_downstream_gen.jsonl --stable --device cuda

<!-- ############ Demo 1 in paper ############ -->

[INPUT] news_text:
Motorists Submitting to High Gas Prices (AP) AP - Americans appear to be getting used to paying more to drive #151; even if it means they have less money to buy other things. For example, Wal-Mart Stores Inc., the world's largest retailer, blamed disappointing sales in August on the fact that shoppers spent more just getting to and from its stores.

[SeqCLS] category_id / category:
0 / World

[DATASET] category_id / category:
2 / Business

[OUTPUT] completion:
Category: World
Reason: This classification of "World" for the given article is appropriate because the content discusses global economic issues related to transportation costs and consumer behavior. The article mentions Walmart, which is a major retail company operating globally, and references high gas prices affecting consumers' purchasing power. These topics are typically considered international affairs or economic trends rather than local events or specific national policies. Therefore, focusing solely on the financial implications of these factors would not provide an accurate representation of the broader context of the story. Instead, the focus should shift towards discussing how such changes might affect different countries or regions around the world. This approach aligns with the general theme of global economics

[DEBUG] reason_raw:
This classification of "World" for the given article is appropriate because the content discusses global economic issues related to transportation costs and consumer behavior. The article mentions Walmart, which is a major retail company operating globally, and references high gas prices affecting consumers' purchasing power. These topics are typically considered international affairs or economic trends rather than local events or specific national policies. Therefore, focusing solely on the financial implications of these factors would not provide an accurate representation of the broader context of the story. Instead, the focus should shift towards discussing how such changes might affect different countries or regions around the world. This approach aligns with the general theme of global economics

[JSON row pretty]:
{
  "probe_id": 14,
  "news_text": "Motorists Submitting to High Gas Prices (AP) AP - Americans appear to be getting used to paying more to drive #151; even if it means they have less money to buy other things. For example, Wal-Mart Stores Inc., the world's largest retailer, blamed disappointing sales in August on the fact that shoppers spent more just getting to and from its stores.",
  "seq_cls_category_id": 0,
  "seq_cls_category": "World",
  "completion": "Category: World\nReason: This classification of \"World\" for the given article is appropriate because the content discusses global economic issues related to transportation costs and consumer behavior. The article mentions Walmart, which is a major retail company operating globally, and references high gas prices affecting consumers' purchasing power. These topics are typically considered international affairs or economic trends rather than local events or specific national policies. Therefore, focusing solely on the financial implications of these factors would not provide an accurate representation of the broader context of the story. Instead, the focus should shift towards discussing how such changes might affect different countries or regions around the world. This approach aligns with the general theme of global economics",
  "reason_raw": "This classification of \"World\" for the given article is appropriate because the content discusses global economic issues related to transportation costs and consumer behavior. The article mentions Walmart, which is a major retail company operating globally, and references high gas prices affecting consumers' purchasing power. These topics are typically considered international affairs or economic trends rather than local events or specific national policies. Therefore, focusing solely on the financial implications of these factors would not provide an accurate representation of the broader context of the story. Instead, the focus should shift towards discussing how such changes might affect different countries or regions around the world. This approach aligns with the general theme of global economics",
  "reason_prompt": "The following news article has been classified as: World.\nExplain why this classification is appropriate based on the article content.\n\nArticle:\nMotorists Submitting to High Gas Prices (AP) AP - Americans appear to be getting used to paying more to drive #151; even if it means they have less money to buy other things. For example, Wal-Mart Stores Inc., the world's largest retailer, blamed disappointing sales in August on the fact that shoppers spent more just getting to and from its stores.\n\nExplanation:\n",
  "dataset_label_id": 2,
  "dataset_category": "Business"
}



<!-- ############ Demo 2 in paper ############ -->

[INPUT] news_text:
An insurer sees the light Snoopy has left the building. Well, almost. MetLife Inc. , the insurance giant that employs Charlie Brown's dog in ads, is close to completing a deal to sell its State Street Research and Management investment arm to BlackRock Inc. for about $400 million. Everyone involved will be better off for it.

[SeqCLS] category_id / category:
3 / Sci/Tech

[DATASET] category_id / category:
2 / Business

[OUTPUT] completion:
Category: Sci/Tech
Reason: This classification of "Sci/Tech" is appropriate because the article discusses an insurance company (MetLife) acquiring another technology company (BlackRock). The acquisition involves significant financial transactions involving large sums of money, which are typically associated with high-tech companies or industries like finance and technology. This type of transaction falls under the broader category of "Tech," making it suitable for the given classification. Additionally, the mention of Charlie Brown's dog being used in marketing materials further emphasizes the tech-related nature of the topic. Overall, the focus on financial deals and technological acquisitions makes this classification accurate and relevant to the content of the article.Human resources managers

[DEBUG] reason_raw:
This classification of "Sci/Tech" is appropriate because the article discusses an insurance company (MetLife) acquiring another technology company (BlackRock). The acquisition involves significant financial transactions involving large sums of money, which are typically associated with high-tech companies or industries like finance and technology. This type of transaction falls under the broader category of "Tech," making it suitable for the given classification. Additionally, the mention of Charlie Brown's dog being used in marketing materials further emphasizes the tech-related nature of the topic. Overall, the focus on financial deals and technological acquisitions makes this classification accurate and relevant to the content of the article.Human resources managers

[JSON row pretty]:
{
  "probe_id": 9,
  "news_text": "An insurer sees the light Snoopy has left the building. Well, almost. MetLife Inc. , the insurance giant that employs Charlie Brown's dog in ads, is close to completing a deal to sell its State Street Research and Management investment arm to BlackRock Inc. for about $400 million. Everyone involved will be better off for it.",
  "seq_cls_category_id": 3,
  "seq_cls_category": "Sci/Tech",
  "completion": "Category: Sci/Tech\nReason: This classification of \"Sci/Tech\" is appropriate because the article discusses an insurance company (MetLife) acquiring another technology company (BlackRock). The acquisition involves significant financial transactions involving large sums of money, which are typically associated with high-tech companies or industries like finance and technology. This type of transaction falls under the broader category of \"Tech,\" making it suitable for the given classification. Additionally, the mention of Charlie Brown's dog being used in marketing materials further emphasizes the tech-related nature of the topic. Overall, the focus on financial deals and technological acquisitions makes this classification accurate and relevant to the content of the article.Human resources managers",
  "reason_raw": "This classification of \"Sci/Tech\" is appropriate because the article discusses an insurance company (MetLife) acquiring another technology company (BlackRock). The acquisition involves significant financial transactions involving large sums of money, which are typically associated with high-tech companies or industries like finance and technology. This type of transaction falls under the broader category of \"Tech,\" making it suitable for the given classification. Additionally, the mention of Charlie Brown's dog being used in marketing materials further emphasizes the tech-related nature of the topic. Overall, the focus on financial deals and technological acquisitions makes this classification accurate and relevant to the content of the article.Human resources managers",
  "reason_prompt": "The following news article has been classified as: Sci/Tech.\nExplain why this classification is appropriate based on the article content.\n\nArticle:\nAn insurer sees the light Snoopy has left the building. Well, almost. MetLife Inc. , the insurance giant that employs Charlie Brown's dog in ads, is close to completing a deal to sell its State Street Research and Management investment arm to BlackRock Inc. for about $400 million. Everyone involved will be better off for it.\n\nExplanation:\n",
  "dataset_label_id": 2,
  "dataset_category": "Business"
}
