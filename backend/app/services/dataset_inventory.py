import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import xml.etree.ElementTree as ET
from dataclasses import asdict

from app.services.metadata import build_product_metadata, ProductMetadata

class DatasetInventory:
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.total_discovered = 0
        self.total_supported = 0
        self.total_unsupported = 0
        self.total_failed = 0
        self.products: List[Dict[str, Any]] = []

    def scan(self):
        # We will map XML labels to their corresponding data files.
        # Often they share the same base name.
        files_by_stem = {}
        for path in self.root_dir.rglob("*"):
            if not path.is_file():
                continue
            self.total_discovered += 1
            
            ext = path.suffix.lower()
            stem = path.stem
            if stem not in files_by_stem:
                files_by_stem[stem] = {'xml': None, 'data': None, 'other': []}
                
            if ext == '.xml':
                files_by_stem[stem]['xml'] = path
            elif ext in ('.img', '.tif', '.tiff', '.png', '.jpg', '.jpeg'):
                files_by_stem[stem]['data'] = path
            else:
                files_by_stem[stem]['other'].append(path)
                
        # Now process the discovered files
        for stem, group in files_by_stem.items():
            xml_path = group['xml']
            data_path = group['data']
            
            if xml_path is None and data_path is None:
                self.total_unsupported += len(group['other'])
                continue

            # We have a supported product
            self.total_supported += (1 if xml_path else 0) + (1 if data_path else 0)
            
            pds4_xml = None
            if xml_path:
                try:
                    with open(xml_path, 'r', encoding='utf-8') as f:
                        pds4_xml = f.read()
                except Exception as e:
                    pass
            
            image_data = b""
            filename = ""
            if data_path:
                filename = data_path.name
                # Note: We do NOT load the full raster into memory here.
                # The inventory should be lightweight.
                # However, our existing build_product_metadata expects image_data for geotiff parsing.
                # Since we don't have GDAL, we can pass an empty bytes or a small header.
                pass
            else:
                filename = xml_path.name if xml_path else ""

            directory_name = data_path.parent.name if data_path else (xml_path.parent.name if xml_path else "")
            
            # Check for non-XML labels if xml_path is none but there's a label file
            label_content = ""
            for o in group['other']:
                if o.suffix.lower() in ('.lbl', '.txt', '.json'):
                    try:
                        with open(o, 'r', encoding='utf-8', errors='ignore') as lf:
                            label_content = lf.read()
                    except:
                        pass

            pm = build_product_metadata(
                filename=filename, 
                image_data=image_data, 
                pds4_xml=pds4_xml,
                directory_name=directory_name,
                label_content=label_content
            )
            pm_dict = pm.to_dict()
            pm_dict['inventory_info'] = {
                'xml_path': str(xml_path) if xml_path else None,
                'data_path': str(data_path) if data_path else None,
                'stem': stem
            }
            self.products.append(pm_dict)

        # Detect duplicates
        from collections import defaultdict
        product_id_map = defaultdict(list)
        for i, prod in enumerate(self.products):
            pid = prod.get('identity', {}).get('product_id')
            if pid:
                product_id_map[pid].append(i)

        duplicate_group_counter = 1
        for pid, indices in product_id_map.items():
            if len(indices) > 1:
                group_id = f"dup_group_{duplicate_group_counter}"
                duplicate_group_counter += 1
                for idx in indices:
                    self.products[idx]['duplicate_group_id'] = group_id

    def get_summary(self) -> Dict[str, Any]:
        return {
            'total_discovered': self.total_discovered,
            'total_supported': self.total_supported,
            'total_unsupported': self.total_unsupported,
            'total_failed': self.total_failed,
            'product_count': len(self.products)
        }
