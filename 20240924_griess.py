import pandas as pd
import numpy as np
import glob

class GreissAssay:
    def __init__(self, filepath_540, filepath_900):
        self.filepath_540 = filepath_540
        self.filepath_900 = filepath_900
        self.data_540 = None
        self.data_900 = None
        self.outliers = None

    def load_data(self):
        """Load 540 nm and 900 nm data from the respective CSV files"""
        self.data_540 = self._process_csv(self.filepath_540)
        self.data_900 = self._process_csv(self.filepath_900)

    def _process_csv(self, filepath):
        """Extracts rows 6, 7, 10, 11, 14, 15, ... and columns 0 and 1, and returns as a NumPy array."""
        # Load the data
        df = pd.read_csv(filepath, header=None)

        # Get the number of rows in the DataFrame
        num_rows = df.shape[0]

        # Generate row indices for rows 6, 7, 10, 11, 14, 15, ...
        rows_to_extract = []
        for n in range((num_rows - 6) // 4 + 1):
            rows_to_extract.append(6 + 4 * n)
            rows_to_extract.append(7 + 4 * n)

        # Extract only the specified columns (0 and 1) and the rows
        extracted_df = df.loc[rows_to_extract, [0, 1]]

        # Convert the extracted data to float values using pd.to_numeric
        extracted_df[0] = pd.to_numeric(extracted_df[0], errors='coerce')
        extracted_df[1] = pd.to_numeric(extracted_df[1], errors='coerce')

        # Convert the extracted data into a NumPy array
        extracted_array = extracted_df.to_numpy(dtype=float)

        return extracted_array


    def detect_outliers_900(self):
        """Detects outliers in data_900."""
        if self.data_900 is None:
            raise ValueError("Data for 900 nm not loaded. Please call load_data() first.")

        # Flatten the data_900 array to apply quartile calculations
        flat_data_900 = self.data_900.flatten()

        # Calculate the 25th and 75th percentiles (q25 and q75)
        q25 = np.percentile(flat_data_900, 25)
        q75 = np.percentile(flat_data_900, 75)

        # Calculate the outlier thresholds
        upper_threshold = q75 + 1.5 * (q75 - q25)
        lower_threshold = q25 - 1.5 * (q75 - q25)

        # Create a boolean mask where True indicates outliers
        mask = (flat_data_900 > upper_threshold) | (flat_data_900 < lower_threshold)

        return mask


    def compute_averages_540(self):
        """
        Compute the average of 540 nm data, ignoring outliers in the corresponding 900 nm sections.
        Take the average of every two rows (rows 1 and 2 give one average, and so on).
        """
        if self.data_540 is None:
            raise ValueError("Data for 540 nm not loaded. Please call load_data() first.")

        # Get the mask for valid data from the 900 nm analysis
        mask = self.detect_outliers_900()

        # Apply the mask to the 540 nm data (ignoring outliers by setting them to NaN)
        flat_data_540 = self.data_540.flatten()
        flat_data_540[mask] = np.nan

        reshaped_array = flat_data_540.reshape(-1, 4)
        averages_540 = np.empty(reshaped_array.shape[0])

        # Iterate over each row in the reshaped array and calculate the mean, handling NaNs
        for i, row in enumerate(reshaped_array):
            if np.all(np.isnan(row)):
                # If all values are NaN, return NaN
                averages_540[i] = np.nan
            else:
                # Otherwise, compute the mean ignoring NaN
                averages_540[i] = np.nanmean(row)

        output_filepath = filepath_540.replace('.CSV', '_average.csv')
        np.savetxt(output_filepath, averages_540, delimiter=",", fmt="%f")

        return averages_540

if __name__ == '__main__':
    # filepath_540 = 'data/20240914_Ik_NO2_standard_540.csv'
    # filepath_900 = 'data/20240914_Ik_NO2_standard_900.csv'
    # griess = GreissAssay(filepath_540, filepath_900)
    # griess.load_data()
    # griess.compute_averages_540()

    patterns = ['data/*_Ik_NO2_time*_540.CSV', 'data/*_Ik_NO2NO3_time*_540.CSV']

    for pattern in patterns:
        filepaths_540 = glob.glob(pattern)

        for filepath_540 in filepaths_540:
            filepath_900 = filepath_540.replace('540', '900')

            griess = GreissAssay(filepath_540, filepath_900)
            griess.load_data()
            griess.compute_averages_540()

