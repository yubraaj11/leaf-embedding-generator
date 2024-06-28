from services.milvus_service import Milvus

ptod= Milvus()

if __name__ == "__main__":
    base_path= "/home/anish/Documents/vertex_project/yub_image_embed/leaf-embedding-generator/dataset"
    ptod.push_to_database(base_path=base_path)