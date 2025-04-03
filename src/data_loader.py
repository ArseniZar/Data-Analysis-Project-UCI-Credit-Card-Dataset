from pathlib import Path
import pandas as pd
from typing import Optional

class DataLoader:
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)  
        self.data: Optional[pd.DataFrame] = self.load_data()

    def load_data(self) -> Optional[pd.DataFrame]:
        try:
            if not self.file_path.exists():
                raise FileNotFoundError(f"File {self.file_path} not found")
            
            self.data = pd.read_csv(
                self.file_path,
                on_bad_lines='warn', 
                encoding_errors='replace'
            )
            
            if self.data.empty:
                return None
                
            return self.data
            
        except FileNotFoundError as e:
            print(f"❌ Error: {e}")
            return None
            
        except pd.errors.EmptyDataError:
            print(f"❌ Error: File {self.file_path} contains no data")
            return None
            
        except pd.errors.ParserError as e:
            print(f"❌ CSV Parsing Error: {e}")
            return None
            
        except Exception as e:
            print(f"❌ Unexpected Error: {e}")
            return None
