"""
The following wrapper classes
allow to use `PynPoint <https://pynpoint.readthedocs.io/en/latest/>`__ with
applefy. PynPoint is not on the requirement list of
applefy. It has to be installed separately.
"""

import sys, os
import shutil
import warnings
from typing import List, Dict
from pathlib import Path


# =============================================================================
# Temporary addition of sys.path.append since pynpoint and applefy were not installed as a package
# =============================================================================
sys.path.append("/Users/Gian/Documents/Github/Pynpoint_ifs/background_files")
sys.path.append('/Users/Gian/Documents/GitHub/Pynpoint')
sys.path.append("/Users/Gian/Documents/Github/applefy")

sys.path.append("C:/Users/BIgsp/Documents/GitHub/Pynpoint_ifs/background_files")
sys.path.append("C:/Users/BIgsp/Documents/GitHub/Pynpoint_ifs")
sys.path.append("C:/Users/BIgsp/Documents/GitHub/applefy")


# =============================================================================
from applefy.detections.contrast import DataReductionInterface


# =============================================================================
# This has to be added once in order to obtain pynpoint correctly
# sys.path.remove('')# removes current directory to prevent wrappers/pynpoint to be used
# =============================================================================

import h5py
import numpy as np

from pynpoint import Pypeline, WavelengthReadingModule, FitsReadingModule, FitCenterModule
from pynpoint import MultiChannelReader, ShiftImagesModule, PaddingModule, DataCubeReplacer
from IFS_Plot import PlotCenterDependantWavelength, PlotSpectrum

from IFS_SimpleSubtraction import IFS_normalizeSpectrum, IFS_binning, IFS_collapseBins, IFS_ClassicalRefSubstraction
from IFS_PCASubtraction import IFS_PCASubtraction

from IFS_SimpleSubtraction import IFS_normalizeSpectrum, IFS_ClassicalRefSubstraction, IFS_binning, IFS_collapseBins

from center_guess import StarCenterFixedGauss, IFS_RefStarAlignment
from jwstframeselection import SelectWavelengthCenterModuleJWST
from IFS_Centering import IFS_Centering



class JWSTSimpleSubtractionPynPoint(DataReductionInterface):
    """
    The JWSTSimpleSubtractionPynPoint is a wrapper around the Simple Subtraction
    implemented in `Pynpoint_ifs`__. 
    """

    def __init__(
            self,
            scratch_dir: Path):
        """
        Constructor of the class.

        Args:
            num_pcas: List of the number of PCA components to be used.
            scratch_dir: A directory to store the Pynpoint database. Any
                Pynpoint database created during the computation will be deleted
                afterwards.
            num_cpus_pynpoint: Number of CPU cores used by Pynpoint.
        """

        self.scratch_dir = scratch_dir
    
    def get_method_keys(self) -> List[str]:
        """
        Get the method name "PCA (#num_pca components)".

        Returns:
            A list with one string "PCA (#num_pca components)".
        """

        return "Residuals(methodsnotimplemented)"

    def __call__(
            self,
            stack_with_fake_planet: np.array,
            stack_dir: str,
            psf_template: str,
            parang_rad,
            exp_id: str
    ) -> np.ndarray:
        """
        Compute the full-frame PCA for several numbers of PCA components.

        Args:
            stack_with_fake_planet: A str containing the path to a fits file.
                Fake plants are inserted by applefy in advance.
            psf_template: A path str to a 2d array fits file with the psf-template
                (usually the unsaturated star).
            parang_rad: is an obselete variable required to be compatible with applefy
            exp_id: Experiment ID of the config used to add the fake
                planet. It is a unique string and can be used to store
                intermediate results. See :meth:`~applefy.detections.\
preparation.generate_fake_planet_experiments` for more information about the
                config files.

        Returns:
            The residual.
        """

        pynpoint_dir = "tmp_pynpoint_" + exp_id
        pynpoint_dir = Path(self.scratch_dir + "/" + pynpoint_dir)

        if not pynpoint_dir.is_dir():
            pynpoint_dir.mkdir()

        out_file = h5py.File(
            pynpoint_dir / "PynPoint_database.hdf5",
            mode='w')

        out_file.create_dataset("data_with_planet", data=stack_with_fake_planet)
        # add header information to data_set?
        out_file.close()

        # 6.) Create PynPoint Pipeline and run PCA
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Disable all print messages from pynpoint
            print("Running SimpleSubtraction... (this may take a while)")
            print("The following steps are performed: \n MultiChannelReader  \
                   \n PaddingModule \
                   \n IFS_normalizeSpectrum \n IFS_binning \
                   \n IFS_RefStarAlignment \n IFS_ClassicalRefSubtraction ")
            sys.stdout = open(os.devnull, 'w')

            pipeline = Pypeline(working_place_in=str(pynpoint_dir),
                                input_place_in=str(pynpoint_dir),
                                output_place_in=str(pynpoint_dir))

            self.set_config_file()
            
            # TODO: cleanup
            #if type(stack_with_fake_planet) is np.ndarray:
            #     cry
                
            # if type(psf_template) is np.ndarray:
            #     also cry
                
                
            # nframes = int(self.m_image_in_port.get_attribute("NFRAMES")[0])

            # wavelengths = self.m_image_in_port.get_attribute('WAV_ARR')[0].astype(np.float16)
            # pixelscale = self.m_image_in_port.get_attribute('PIXSCALE')[0].astype(np.float16)
            # bands = self.m_image_in_port.get_attribute('BAND_ARR')[0].astype(str)
            
            # =============================================================================
            # Set science frame
            # =============================================================================


            Import_Science = MultiChannelReader(name_in = "readersci",
                                        input_dir = stack_dir,
                                        image_tag = "sci",
                                        check = True,
                                        ifs_data=False,
                                        overwrite=True)

            pipeline.add_module(Import_Science)
            pipeline.run_module("readersci")
            
            
            # =============================================================================
            # Set ref frame
            # =============================================================================
            
            
            Import_Science = MultiChannelReader(name_in = "readerref",
                                        input_dir = psf_template,
                                        image_tag = "ref",
                                        check = True,
                                        ifs_data=False,
                                        overwrite=True)

            pipeline.add_module(Import_Science)
            pipeline.run_module("readerref")
            
            # =============================================================================
            #  narrowing wavelength and binning           
            # =============================================================================
            
            selection = SelectWavelengthCenterModuleJWST(name_in="selective",
                                                         image_in_tag="sci",
                                                         image_out_tag="select_sci",
                                                         nr_frames = 10,
                                                         wave_center = 6.0)
            pipeline.add_module(selection)
            pipeline.run_module("selective")

            binning = IFS_collapseBins(name_in="binnion",
                                       image_in_tag="select_sci",
                                       image_out_tag="bin_sci")

            pipeline.add_module(binning)
            pipeline.run_module("binnion")
            
            ReplacePlanets = DataCubeReplacer(name_in="replacer",
                                              image_in_tag="bin_sci",
                                              image_out_tag="new_sci",
                                              new_cube=stack_with_fake_planet)
            pipeline.add_module(ReplacePlanets)
            pipeline.run_module("replacer")
            
            
            selectionref = SelectWavelengthCenterModuleJWST(name_in="selectiveref",
                                                         image_in_tag="ref",
                                                         image_out_tag="select_ref",
                                                         nr_frames = 10,
                                                         wave_center = 6.0)
            pipeline.add_module(selectionref)
            pipeline.run_module("selectiveref")

            binningref = IFS_collapseBins(name_in="binnionref",
                                          image_in_tag="select_ref",
                                          image_out_tag="bin_ref")

            pipeline.add_module(binningref)
            pipeline.run_module("binnionref")
            
                        
            # =============================================================================
            # Bring reference and science to same shape by Padding
            # =============================================================================


            paddington = PaddingModule(name_in="pad",
                                       image_in_tags=["new_sci","bin_ref"],
                                       image_out_suff="pad")

            pipeline.add_module(paddington)
            pipeline.run_module("pad")
            
            # =============================================================================
            # Center and Normalize science target
            # =============================================================================


            # dat = pipeline.get_data("new_sci_pad")

            # shift = StarCenterFixedGauss(dat)

            # module = ShiftImagesModule(name_in='shift',
            #                                     image_in_tag='new_sci_pad',
            #                                     shift_xy=shift,
            #                                     image_out_tag='centered')
            # pipeline.add_module(module)
            # pipeline.run_module("shift")
            
            module = IFS_Centering(name_in = "centermod",
                                   image_in_tag = "new_sci_pad",
                                   fit_out_tag = "shift")

            pipeline.add_module(module)
            pipeline.run_module("centermod")

            module = ShiftImagesModule(name_in='shifter',
                                                image_in_tag='new_sci_pad',
                                                shift_xy="shift",
                                                image_out_tag='centered')
            pipeline.add_module(module)
            pipeline.run_module("shifter")

            module = IFS_normalizeSpectrum(name_in='norm',
                                           image_in_tag='centered',
                                           image_out_tag='normed')
            pipeline.add_module(module)
            pipeline.run_module("norm")
            
            # =============================================================================
            # Center and Normalize ref target
            # =============================================================================

            # dat_ref = pipeline.get_data("ref_pad")

            # shift_ref = StarCenterFixedGauss(dat_ref)

            # module = ShiftImagesModule(name_in='shift_ref',
            #                                     image_in_tag='ref_pad',
            #                                     shift_xy=shift_ref,
            #                                     image_out_tag='centered_ref')
            # pipeline.add_module(module)
            # pipeline.run_module("shift_ref")
            
            module = IFS_Centering(name_in = "centermod_ref",
                                   image_in_tag = "bin_ref_pad",
                                   fit_out_tag = "shift_ref")

            pipeline.add_module(module)
            pipeline.run_module("centermod_ref")

            module = ShiftImagesModule(name_in='shifter_ref',
                                                image_in_tag='bin_ref_pad',
                                                shift_xy="shift_ref",
                                                image_out_tag='centered_ref')
            pipeline.add_module(module)
            pipeline.run_module("shifter_ref")

            module = IFS_normalizeSpectrum(name_in='norm_ref',
                                           image_in_tag='centered_ref',
                                           image_out_tag='normed_ref')
            pipeline.add_module(module)
            pipeline.run_module("norm_ref")
            

            


            # # =============================================================================
            # # Align to science star
            # # =============================================================================

            # test = IFS_RefStarAlignment(name_in="aligner", 
            #                             sci_in_tag="normed", 
            #                             ref_in_tags="normed_ref", 
            #                             fit_out_tag_suff="opt",
            #                             qual_method = "L2",
            #                             in_rad = 0.1,
            #                             out_rad = 2.5,
            #                             apertshap = "Circle")
            # pipeline.add_module(test)
            # pipeline.run_module("aligner")
            
            # =============================================================================
            # Residual
            # =============================================================================


            resid_calc = IFS_ClassicalRefSubstraction(name_in='residual',
                                                      image_in_tags=['normed','normed_ref'],
                                                      image_out_tag='Residual')

            pipeline.add_module(resid_calc)
            pipeline.run_module("residual")
            
            


    # TODO: properly output results! and check if header info needs to be saved as well

            # 7.) Get the data from the Pynpoint database
            result_dict = dict()

            residuals = pipeline.get_data("Residual")
            result_dict["Residual"] = residuals

            # Delete the temporary database
            shutil.rmtree(pynpoint_dir)

            # Enable print messages again
            sys.stdout = sys.__stdout__
            print("All PynPoint Modules have run successfully. Finishing up...")

        return result_dict
    
    def set_config_file(self):
        import configparser
        config = configparser.ConfigParser()
        config.add_section('header')
        config['header']['INSTRUMENT'] = 'INSTRUME'
        config['header']['DIT'] = 'EFFINTTM'
        config['header']['NDIT'] = 'NINTS'
        config['header']['NFRAMES'] = 'NAXIS3'
        config['header']['NAXISA'] = 'NAXIS1'
        config['header']['NAXISB'] = 'NAXIS2'
        config['header']['DITHER_X'] = 'XOFFSET'
        config['header']['DITHER_Y'] = 'YOFFSET'
        config['header']['DATE'] = 'DATE-OBS'
        config['header']['RA'] = 'TARG_RA'
        config['header']['DEC'] = 'TARG_DEC'
        config['header']['WAV_START'] = 'CRVAL3'
        config['header']['WAV_INCR'] = 'CDELT3'
        
        config.add_section('settings')
        config['settings']['PIXSCALE'] = '0.13'
        config['settings']['MEMORY'] = 'None'
        config['settings']['CPU'] = '1'

        with open('PynPoint_config.ini', 'w') as configfile:
            config.write(configfile)


class JWSTPCASubtractionPynPoint(DataReductionInterface):
    """
    The JWSTPCASubtractionPynPoint is a wrapper around the PCA Subtraction
    implemented in `Pynpoint_ifs`__.
    """

    def __init__(
            self,
            num_pcas: List[int],
            scratch_dir: Path,
            num_cpus_pynpoint: int = 1):
        """
        Constructor of the class.

        Args:
            num_pcas: List of the number of PCA components to be used.
            scratch_dir: A directory to store the Pynpoint database. Any
                Pynpoint database created during the computation will be deleted
                afterwards.
            num_cpus_pynpoint: Number of CPU cores used by Pynpoint.
        """

        self.num_pcas = num_pcas
        self.scratch_dir = scratch_dir
        self.num_cpus_pynpoint = num_cpus_pynpoint

    def get_method_keys(self) -> List[str]:
        """
        Get the method name "PCA (#num_pca components)".

        Returns:
            A list with one string "PCA (#num_pca components)".
        """

        keys = ["PCA (" + str(num_pcas).zfill(3) + " components)"
                for num_pcas in self.num_pcas]

        return keys

    def __call__(
            self,
            stack_with_fake_planet: np.array,
            ref_dir: str,
            psf_template: str,
            parang_rad,
            exp_id: str
    ) -> np.ndarray:
        """
        Compute the full-frame PCA for several numbers of PCA components.

        Args:
            stack_with_fake_planet: A str containing the path to a fits file.
                Fake plants are inserted by applefy in advance.
            psf_template: A path str to a 2d array fits file with the psf-template
                (usually the unsaturated star).
            parang_rad: is an obselete variable required to be compatible with applefy
            exp_id: Experiment ID of the config used to add the fake
                planet. It is a unique string and can be used to store
                intermediate results. See :meth:`~applefy.detections.\
preparation.generate_fake_planet_experiments` for more information about the
                config files.

        Returns:
            The residual.
        """

        pynpoint_dir = "tmp_pynpoint_" + exp_id
        pynpoint_dir = Path(self.scratch_dir + "/" + pynpoint_dir)

        if not pynpoint_dir.is_dir():
            pynpoint_dir.mkdir()

        out_file = h5py.File(
            pynpoint_dir / "PynPoint_database.hdf5",
            mode='w')

        out_file.create_dataset("data_with_planet", data=stack_with_fake_planet)
        # add header information to data_set?
        out_file.close()

        # 6.) Create PynPoint Pipeline and run PCA
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Disable all print messages from pynpoint
            print("Running SimpleSubtraction... (this may take a while)")
            print("The following steps are performed: \n MultiChannelReader  \
                   \n PaddingModule \
                   \n IFS_normalizeSpectrum \n IFS_binning \
                   \n IFS_RefStarAlignment \n IFS_ClassicalRefSubtraction ")
            sys.stdout = open(os.devnull, 'w')

            pipeline = Pypeline(working_place_in=str(pynpoint_dir),
                                input_place_in=str(pynpoint_dir),
                                output_place_in=str(pynpoint_dir))

            self.set_config_file()

            # TODO: cleanup
            # if type(stack_with_fake_planet) is np.ndarray:
            #     cry

            # if type(psf_template) is np.ndarray:
            #     also cry

            # nframes = int(self.m_image_in_port.get_attribute("NFRAMES")[0])

            # wavelengths = self.m_image_in_port.get_attribute('WAV_ARR')[0].astype(np.float16)
            # pixelscale = self.m_image_in_port.get_attribute('PIXSCALE')[0].astype(np.float16)
            # bands = self.m_image_in_port.get_attribute('BAND_ARR')[0].astype(str)

            # =============================================================================
            # Set science frame
            # =============================================================================

            Import_Science = MultiChannelReader(name_in="readersci",
                                                input_dir=stack_dir,
                                                image_tag="sci",
                                                check=True,
                                                ifs_data=False,
                                                overwrite=True)

            pipeline.add_module(Import_Science)
            pipeline.run_module("readersci")

            # =============================================================================
            # Set ref frame
            # =============================================================================

            Import_Science = MultiChannelReader(name_in="readerref",
                                                input_dir=psf_template,
                                                image_tag="ref",
                                                check=True,
                                                ifs_data=False,
                                                overwrite=True)

            pipeline.add_module(Import_Science)
            pipeline.run_module("readerref")

            # =============================================================================
            #  narrowing wavelength and binning
            # =============================================================================

            selection = SelectWavelengthCenterModuleJWST(name_in="selective",
                                                         image_in_tag="sci",
                                                         image_out_tag="select_sci",
                                                         nr_frames=10,
                                                         wave_center=6.0)
            pipeline.add_module(selection)
            pipeline.run_module("selective")

            binning = IFS_collapseBins(name_in="binnion",
                                       image_in_tag="select_sci",
                                       image_out_tag="bin_sci")

            pipeline.add_module(binning)
            pipeline.run_module("binnion")

            ReplacePlanets = DataCubeReplacer(name_in="replacer",
                                              image_in_tag="bin_sci",
                                              image_out_tag="new_sci",
                                              new_cube=stack_with_fake_planet)
            pipeline.add_module(ReplacePlanets)
            pipeline.run_module("replacer")

            selectionref = SelectWavelengthCenterModuleJWST(name_in="selectiveref",
                                                            image_in_tag="ref",
                                                            image_out_tag="select_ref",
                                                            nr_frames=10,
                                                            wave_center=6.0)
            pipeline.add_module(selectionref)
            pipeline.run_module("selectiveref")

            binningref = IFS_collapseBins(name_in="binnionref",
                                          image_in_tag="select_ref",
                                          image_out_tag="bin_ref")

            pipeline.add_module(binningref)
            pipeline.run_module("binnionref")

            # =============================================================================
            # Bring reference and science to same shape by Padding
            # =============================================================================

            paddington = PaddingModule(name_in="pad",
                                       image_in_tags=["new_sci", "bin_ref"],
                                       image_out_suff="pad")

            pipeline.add_module(paddington)
            pipeline.run_module("pad")

            # =============================================================================
            # Center and Normalize science target
            # =============================================================================

            # dat = pipeline.get_data("new_sci_pad")

            # shift = StarCenterFixedGauss(dat)

            # module = ShiftImagesModule(name_in='shift',
            #                                     image_in_tag='new_sci_pad',
            #                                     shift_xy=shift,
            #                                     image_out_tag='centered')
            # pipeline.add_module(module)
            # pipeline.run_module("shift")

            module = IFS_Centering(name_in="centermod",
                                   image_in_tag="new_sci_pad",
                                   fit_out_tag="shift")

            pipeline.add_module(module)
            pipeline.run_module("centermod")

            module = ShiftImagesModule(name_in='shifter',
                                       image_in_tag='new_sci_pad',
                                       shift_xy="shift",
                                       image_out_tag='centered')
            pipeline.add_module(module)
            pipeline.run_module("shifter")

            module = IFS_normalizeSpectrum(name_in='norm',
                                           image_in_tag='centered',
                                           image_out_tag='normed')
            pipeline.add_module(module)
            pipeline.run_module("norm")

            # =============================================================================
            # Center and Normalize ref target
            # =============================================================================

            # dat_ref = pipeline.get_data("ref_pad")

            # shift_ref = StarCenterFixedGauss(dat_ref)

            # module = ShiftImagesModule(name_in='shift_ref',
            #                                     image_in_tag='ref_pad',
            #                                     shift_xy=shift_ref,
            #                                     image_out_tag='centered_ref')
            # pipeline.add_module(module)
            # pipeline.run_module("shift_ref")

            module = IFS_Centering(name_in="centermod_ref",
                                   image_in_tag="bin_ref_pad",
                                   fit_out_tag="shift_ref")

            pipeline.add_module(module)
            pipeline.run_module("centermod_ref")

            module = ShiftImagesModule(name_in='shifter_ref',
                                       image_in_tag='bin_ref_pad',
                                       shift_xy="shift_ref",
                                       image_out_tag='centered_ref')
            pipeline.add_module(module)
            pipeline.run_module("shifter_ref")

            module = IFS_normalizeSpectrum(name_in='norm_ref',
                                           image_in_tag='centered_ref',
                                           image_out_tag='normed_ref')
            pipeline.add_module(module)
            pipeline.run_module("norm_ref")

            # # =============================================================================
            # # Align to science star
            # # =============================================================================

            # test = IFS_RefStarAlignment(name_in="aligner",
            #                             sci_in_tag="normed",
            #                             ref_in_tags="normed_ref",
            #                             fit_out_tag_suff="opt",
            #                             qual_method = "L2",
            #                             in_rad = 0.1,
            #                             out_rad = 2.5,
            #                             apertshap = "Circle")
            # pipeline.add_module(test)
            # pipeline.run_module("aligner")

            # =============================================================================
            # Residual
            # =============================================================================

            resid_calc = IFS_ClassicalRefSubstraction(name_in='residual',
                                                      image_in_tags=['normed', 'normed_ref'],
                                                      image_out_tag='Residual')

            pipeline.add_module(resid_calc)
            pipeline.run_module("residual")

            # TODO: properly output results! and check if header info needs to be saved as well

            # 7.) Get the data from the Pynpoint database
            result_dict = dict()

            residuals = pipeline.get_data("Residual")
            result_dict["Residual"] = residuals

            # Delete the temporary database
            shutil.rmtree(pynpoint_dir)

            # Enable print messages again
            sys.stdout = sys.__stdout__
            print("All PynPoint Modules have run successfully. Finishing up...")

        return result_dict

    def set_config_file(self):
        import configparser
        config = configparser.ConfigParser()
        config.add_section('header')
        config['header']['INSTRUMENT'] = 'INSTRUME'
        config['header']['DIT'] = 'EFFINTTM'
        config['header']['NDIT'] = 'NINTS'
        config['header']['NFRAMES'] = 'NAXIS3'
        config['header']['NAXISA'] = 'NAXIS1'
        config['header']['NAXISB'] = 'NAXIS2'
        config['header']['DITHER_X'] = 'XOFFSET'
        config['header']['DITHER_Y'] = 'YOFFSET'
        config['header']['DATE'] = 'DATE-OBS'
        config['header']['RA'] = 'TARG_RA'
        config['header']['DEC'] = 'TARG_DEC'
        config['header']['WAV_START'] = 'CRVAL3'
        config['header']['WAV_INCR'] = 'CDELT3'

        config.add_section('settings')
        config['settings']['PIXSCALE'] = '0.13'
        config['settings']['MEMORY'] = 'None'
        config['settings']['CPU'] = '1'

        with open('PynPoint_config.ini', 'w') as configfile:
            config.write(configfile)