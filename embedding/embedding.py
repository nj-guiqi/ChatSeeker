import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from pymilvus import MilvusClient, Collection, CollectionSchema, FieldSchema, DataType, Function, FunctionType
from pymilvus import model, RRFRanker, AnnSearchRequest
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import numpy as np
import os

from .logger import db_logger

class MilvusClientWrapper:
    def __init__(self, config):
        self.client = MilvusClient(host=config["milvus_host"], port=config["milvus_port"])
        self.config = config

    def has_collection(self, collection_name: str):
        try:
            has_collection = self.client.has_collection(collection_name)
            return has_collection
        except Exception as e:
            db_logger.error(f"检查集合失败: {e}")
            raise
        

    def create_collection(self, collection_name: str, info_dict: dict):
        try:
            has_collection = self.client.has_collection(collection_name)
            if has_collection:
                db_logger.info(f"collection: {collection_name} exists")
                return 
            schema = self._create_milvus_schema(info_dict)
            # 后续加入设置进行控制，此处用于构建评估数据集
            functions = Function(
                name="bm25_sparse_vector",
                function_type=FunctionType.BM25,
                input_field_names=[self.config["content_key"]],
                output_field_names=["bm25_sparse_vector"],
            )
            schema.add_function(functions)
            self.client.create_collection(collection_name=collection_name, schema=schema)
            self._set_index(collection_name)
            db_logger.info(f"collection: {collection_name} created")
        except Exception as e:
            db_logger.error(f" fail create colleaction: {e}")
            raise

    def _create_milvus_schema(self, info_dict: dict):
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=self.config["vector_dim"]),
            # FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
            FieldSchema(name="bm25_sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR)
        ]
        for key, value in info_dict.items():
            if key == self.config["content_key"]:
                fields.append(FieldSchema(name=key, dtype=DataType.VARCHAR, max_length=5096, is_primary=False,
                                          enable_match=True, enable_analyzer=True))
            elif key == self.config["chunk_id_key"]:
                fields.append(FieldSchema(name=key, dtype=DataType.INT64, is_primary=False, auto_id=False))
            # elif key == self.config["db_metadata_key"]:
            #     fields.append(FieldSchema(name=key, dtype=DataType.JSON, is_primary=False))
            else:
                if isinstance(value, str):
                    fields.append(FieldSchema(name=key, dtype=DataType.VARCHAR, max_length=3072, is_primary=False))
                elif isinstance(value, int):
                    fields.append(FieldSchema(name=key, dtype=DataType.INT64, is_primary=False))
                else:
                    fields.append(FieldSchema(name=key, dtype=DataType.JSON, is_primary=False))
        schema = CollectionSchema(fields=fields, description="conversation collection schema")
        return schema

    def _set_index(self, collection_name: str):
        try:
            index_params = self.client.prepare_index_params()
            for field_name in self.config["db_index_fields"]:
                if field_name == "bm25_sparse_vector":
                    index_params.add_index(
                        field_name=field_name,
                        index_type="SPARSE_INVERTED_INDEX",
                        metric_type="BM25",
                    )
                # 稀疏向量索引
                elif field_name == "sparse_vector":
                    index_params.add_index(
                        field_name=field_name,
                        index_type="SPARSE_INVERTED_INDEX",
                        metric_type="IP",
                        params={"nlist": 1024},
                    )
                # 稠密向量索引
                else:
                    index_params.add_index(
                        field_name=field_name,
                        index_type="IVF_FLAT",
                        metric_type="IP",
                        params={"nlist": 1024},
                        # params={"M": 16, "efConstruction": 100},  # 使用HNSW时
                    )
            self.client.create_index(collection_name=collection_name, index_params=index_params)
            db_logger.info(f"collection: {collection_name} index created")
        except Exception as e:
            db_logger.error(f"fail create index: {e}")
            raise

    # 插入数据
    def insert(self, collection_name: str, data: list[dict], embeddings: list[dict]):
        data = self._insert_preprocess(data, embeddings)
        try:
            res = self.client.insert(collection_name=collection_name, data=data)
            db_logger.info(f"表 {collection_name} 插入数据: {res['insert_count']} 条")
        except Exception as e:
            db_logger.error(f"插入数据失败: {e}")
            raise

    # 插入前encode数据
    def _insert_preprocess(self, data: list[dict], embeddings: list[dict]):
        for i, item in enumerate(data):
            item["dense_vector"] = embeddings[i]["dense"][0]
            # item["sparse_vector"] = embeddings[i]["sparse"]
            # item["colbert_vector"] = embeddings[i]["colbert_vecs"]
            if "id" in item:
                del item["id"]
        return data

    def insert_data(self, chunked_data, collection_name, densemodel):
        # 插入数据并显示进度条
        total_batches = len(chunked_data) // self.config["batch_size"] + (
            1 if len(chunked_data) % self.config["batch_size"] != 0 else 0)
        for i in tqdm(range(total_batches), desc="Inserting data"):
            batch_data = chunked_data[i * self.config["batch_size"]: (i + 1) * self.config["batch_size"]]
            embeddings = []
            for item in batch_data:
                embedding = densemodel.embedding(item[self.config["content_key"]])
                embedding = {
                    'dense': [embedding]
                }
                embeddings.append(embedding)

            self.insert(collection_name, batch_data, embeddings)
            
    # load_data
    def load_data(self, collection_name: str):
        try:
            self.client.load_collection(collection_name=collection_name)
        except Exception as e:
            db_logger.error(f"加载数据失败: {e}")
            raise
    
    # 搜索
    def search(self,
        collection_name: str,
        query_embedding: list,
        search_params: dict,
        limit: int,
        output_fields: list,
        search_field: str,
        **kwargs
    ):
        try:
            results = self.client.search(
                collection_name=collection_name,
                data=[query_embedding],
                output_fields=output_fields,
                anns_field=search_field,
                search_params=search_params,
                limit=limit,
                **kwargs
            )
            return results[0]
        except Exception as e:
            print(f"搜索失败: {e}")