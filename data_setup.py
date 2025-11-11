"""
Data preparation and loading utilities.
"""
import os
import requests
import zipfile
import tarfile
from pathlib import Path
from typing import Optional, Tuple, List

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


class DataDownloader:
    """Handles downloading and extracting datasets."""
    
    @staticmethod
    def download_data(source: str, destination: str, force_download: bool = False) -> Path:
        """
        Download data from a source URL.
        
        Args:
            source: URL to download from
            destination: Directory to save downloaded file
            force_download: Whether to force download even if file exists
            
        Returns:
            Path to downloaded file
        """
        data_path = Path(destination)
        data_path.mkdir(parents=True, exist_ok=True)
        
        filename = Path(source).name
        if not filename:
            filename = "downloaded_file"
            
        local_path = data_path / filename
        
        if not local_path.exists() or force_download:
            print(f"[INFO] Downloading {filename} from {source}...")
            
            response = requests.get(source, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            with open(local_path, 'wb') as f, tqdm(
                total=total_size, unit='B', unit_scale=True, desc=filename
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            print(f"[INFO] Download completed: {local_path}")
        else:
            print(f"[INFO] File already exists: {local_path}")
            
        return local_path
    
    @staticmethod
    def extract_data(file_path: Path, extract_to: Path, remove_source: bool = False) -> Path:
        """
        Extract compressed files.
        
        Args:
            file_path: Path to compressed file
            extract_to: Directory to extract to
            remove_source: Whether to remove source after extraction
            
        Returns:
            Path to extracted content
        """
        extract_to.mkdir(parents=True, exist_ok=True)
        
        print(f"[INFO] Extracting {file_path} to {extract_to}...")
        
        if file_path.suffix == '.zip':
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif file_path.suffix in ['.tar', '.tgz'] or file_path.suffixes[-2:] == ['.tar', '.gz']:
            with tarfile.open(file_path, 'r:*') as tar_ref:
                tar_ref.extractall(extract_to)
        
        print(f"[INFO] Extraction completed: {extract_to}")
        
        if remove_source:
            file_path.unlink()
            print(f"[INFO] Removed source file: {file_path}")
            
        return extract_to
    
    @staticmethod
    def download_and_extract(source: str, destination: str, extract_to: Optional[str] = None, 
                           remove_source: bool = True) -> Path:
        """
        Download and extract data in one step.
        
        Args:
            source: URL to download from
            destination: Directory to save downloaded file
            extract_to: Directory to extract files to (defaults to destination)
            remove_source: Whether to remove source after extraction
            
        Returns:
            Path to extracted data directory
        """
        downloaded_file = DataDownloader.download_data(source, destination)
        extract_path = Path(extract_to) if extract_to else Path(destination)
        
        return DataDownloader.extract_data(downloaded_file, extract_path, remove_source)


class DataLoaderCreator:
    """Creates PyTorch DataLoaders for image classification."""
    
    @staticmethod
    def create_dataloaders(train_dir: str, test_dir: str, transform: transforms.Compose, 
                          batch_size: int = 32, num_workers: int = None) -> Tuple[DataLoader, DataLoader, List[str]]:
        """
        Create training and testing DataLoaders.
        
        Args:
            train_dir: Path to training directory
            test_dir: Path to testing directory
            transform: torchvision transforms to perform on data
            batch_size: Number of samples per batch
            num_workers: Number of workers per DataLoader
            
        Returns:
            Tuple of (train_dataloader, test_dataloader, class_names)
        """
        num_workers = num_workers or os.cpu_count()
        
        # Create datasets
        train_data = datasets.ImageFolder(train_dir, transform=transform)
        test_data = datasets.ImageFolder(test_dir, transform=transform)
        
        # Get class names
        class_names = train_data.classes
        
        # Create DataLoaders
        train_dataloader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        
        test_dataloader = DataLoader(
            test_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        
        return train_dataloader, test_dataloader, class_names
    
    @staticmethod
    def get_image_paths(data_directory: str, extensions: tuple = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')) -> List[Path]:
        """
        Get all image file paths from a directory and its subdirectories.
        
        Args:
            data_directory: Directory to search for images
            extensions: Tuple of image file extensions to include
            
        Returns:
            List of Path objects for all image files
        """
        data_path = Path(data_directory)
        image_paths = []
        
        for ext in extensions:
            image_paths.extend(data_path.rglob(f"*{ext}"))
            image_paths.extend(data_path.rglob(f"*{ext.upper()}"))
        
        print(f"[INFO] Found {len(image_paths)} images in {data_directory}")
        return sorted(image_paths)

    @staticmethod
    def walk_through_dir(dir_path: str):
        """
        Walks through dir_path returning its contents.
        
        Args:
            dir_path (str): target directory
        
        Returns:
            A print out of:
              number of subdirectories in dir_path
              number of images (files) in each subdirectory
              name of each subdirectory
        """
        for dirpath, dirnames, filenames in os.walk(dir_path):
            print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")


def download_data(source: str, destination: str, remove_source: bool = True) -> Path:
    """
    Downloads a zipped dataset from source and unzips to destination.
    
    Args:
        source (str): A link to a zipped file containing data.
        destination (str): A target directory to unzip data to.
        remove_source (bool): Whether to remove the source after downloading and extracting.
    
    Returns:
        pathlib.Path to downloaded data.
    
    Example usage:
        download_data(source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
                      destination="pizza_steak_sushi")
    """
    # Setup path to data folder
    data_path = Path("data/")
    image_path = data_path / destination

    # If the image folder doesn't exist, download it and prepare it... 
    if image_path.is_dir():
        print(f"[INFO] {image_path} directory exists, skipping download.")
    else:
        print(f"[INFO] Did not find {image_path} directory, creating one...")
        image_path.mkdir(parents=True, exist_ok=True)
        
        # Download pizza, steak, sushi data
        target_file = Path(source).name
        with open(data_path / target_file, "wb") as f:
            request = requests.get(source)
            print(f"[INFO] Downloading {target_file} from {source}...")
            f.write(request.content)

        # Unzip pizza, steak, sushi data
        with zipfile.ZipFile(data_path / target_file, "r") as zip_ref:
            print(f"[INFO] Unzipping {target_file} data...") 
            zip_ref.extractall(image_path)

        # Remove .zip file
        if remove_source:
            os.remove(data_path / target_file)
    
    return image_path