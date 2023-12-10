"""
The following wrapper classes
allow to use `PynPoint <https://pynpoint.readthedocs.io/en/latest/>`__ with
applefy. PynPoint is not on the requirement list of
applefy. It has to be installed separately.
"""

import sys, os
import shutil
import warnings
from typing import List, Dict, Optional
from pathlib import Path


# =============================================================================
# Temporary addition of sys.path.append since pynpoint and applefy were not installed as a package
# =============================================================================
sys.path.append("/Users/Gian/Documents/Github/Pynpoint_ifs/background_files")
sys.path.append('/Users/Gian/Documents/GitHub/Pynpoint')
sys.path.append("/Users/Gian/Documents/Github/applefy")
# =============================================================================
from applefy.detections.contrast import DataReductionInterface

sys.path.append("C:/Users/BIgsp/Documents/GitHub/Pynpoint_ifs/background_files")
sys.path.append("C:/Users/BIgsp/Documents/GitHub/Pynpoint_ifs")
sys.path.append("C:/Users/BIgsp/Documents/GitHub/applefy")


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

from center_guess import StarCenterFixedGauss, IFS_RefStarAlignment
from IFS_PCASubtraction import IFS_PCASubtraction
from jwstframeselection import SelectWavelengthCenterModuleJWST
from IFS_Centering import IFS_Centering



class JWSTSimpleSubtractionPynPoint_unopt(DataReductionInterface):
    """
    The JWSTSimpleSubtractionPynPoint is a wrapper around the Simple Subtraction
    implemented in `Pynpoint_ifs`__. 
    """

    def __init__(
            self,
            scratch_dir: Path,
            psf_dir: str,
            psf_list: Optional[List[str]]=None,
            access_pipeline: Optional[bool]=False):
        """
        Constructor of the class.

        Args:
            scratch_dir: A directory to store the Pynpoint database. Any
                Pynpoint database created during the computation will be deleted
                afterwards.
            psf_list: List of psf template directories. If not given only 
                the psf_template inputted in __call__ will be used.
        """

        self.scratch_dir = scratch_dir
        self.psf_list = psf_list
        self.psf_dir = psf_dir
        self.publish_pipeline = access_pipeline
    
    def get_method_keys(self) -> List[str]:
        """
        Get the method name "PCA (#num_pca components)".

        Returns:
            A list with one string "PCA (#num_pca components)".
        """
        
        keys = []
        
        if self.psf_list is not None:
            stars = [end.split('/')[-1] for end in self.psf_list]
            keys = ["Refstar_"+star for star in stars]
        ref_star = self.psf_dir.split('/')[-1]
        keys.insert(0,"MainRefstar_"+ref_star)
        return keys

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
        self.psf_dir = psf_template

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
            
            reference_list = []
            more_stars = False
            
            if self.psf_list is not None:
                more_stars = True
                reference_list = self.psf_list
            
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
            # Set ref library if given
            # =============================================================================
            
            if more_stars:
                stars = [end.split('/')[-1] for end in reference_list]
                
                for star,star_dir in zip(stars,reference_list):
                    import_module = MultiChannelReader(name_in = "reader_"+star,
                                                       input_dir = star_dir,
                                                       image_tag = star,
                                                       check=True,
                                                       ifs_data=False,
                                                       overwrite=True)
                    pipeline.add_module(import_module)
                    pipeline.run_module("reader_"+star)
                    
                
            
            # =============================================================================
            #  narrowing wavelength and binning           
            # =============================================================================
            
            selection = SelectWavelengthCenterModuleJWST(name_in="selective",
                                                         image_in_tag="sci",
                                                         image_out_tag="select_sci",
                                                         nr_frames = 10,
                                                         wave_center = 5.3)
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
                                                         wave_center = 5.3)
            pipeline.add_module(selectionref)
            pipeline.run_module("selectiveref")

            binningref = IFS_collapseBins(name_in="binnionref",
                                          image_in_tag="select_ref",
                                          image_out_tag="bin_ref")

            pipeline.add_module(binningref)
            pipeline.run_module("binnionref")
            
            if more_stars:
                for star in stars:
                    selectionref = SelectWavelengthCenterModuleJWST(name_in="select_"+star,
                                                                 image_in_tag= star,
                                                                 image_out_tag="selected_"+star,
                                                                 nr_frames = 10,
                                                                 wave_center = 5.3)
                    pipeline.add_module(selectionref)
                    pipeline.run_module("select_"+star)

                    binningref = IFS_collapseBins(name_in="binning_"+star,
                                                  image_in_tag="selected_"+star,
                                                  image_out_tag="bin_"+star)

                    pipeline.add_module(binningref)
                    pipeline.run_module("binning_"+star)
            
                        
            # =============================================================================
            # Bring reference and science to same shape by Padding
            # =============================================================================

            if more_stars:
                bin_names = ["bin_"+star for star in stars]
                bin_names.append("new_sci")
                bin_names.append("bin_ref")
                paddington = PaddingModule(name_in="pad",
                                           image_in_tags=bin_names,
                                           image_out_suff="pad")

                pipeline.add_module(paddington)
                pipeline.run_module("pad")
            
            else:
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

            # module = IFS_normalizeSpectrum(name_in='norm',
            #                                image_in_tag='centered',
            #                                image_out_tag='normed')
            # pipeline.add_module(module)
            # pipeline.run_module("norm")
            
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

            # module = IFS_normalizeSpectrum(name_in='norm_ref',
            #                                image_in_tag='centered_ref',
            #                                image_out_tag='normed_ref')
            # pipeline.add_module(module)
            # pipeline.run_module("norm_ref")
            

            # =============================================================================
            #   Center and normalize more ref stars          
            # =============================================================================
            if more_stars:
                for star in stars:
                    module = IFS_Centering(name_in = "centermod_ref"+star,
                                           image_in_tag = "bin_"+star+"_pad",
                                           fit_out_tag = "shift_"+star)

                    pipeline.add_module(module)
                    pipeline.run_module("centermod_ref"+star)

                    module = ShiftImagesModule(name_in='shifter_ref'+star,
                                                        image_in_tag="bin_"+star+"_pad",
                                                        shift_xy="shift_"+star,
                                                        image_out_tag='centered_'+star)
                    pipeline.add_module(module)
                    pipeline.run_module("shifter_ref"+star)
            

            # # =============================================================================
            # # Align to science star
            # # =============================================================================

            test = IFS_RefStarAlignment(name_in="aligner", 
                                        sci_in_tag="centered", 
                                        ref_in_tags="centered_ref", 
                                        fit_out_tag_suff="opt",
                                        qual_method = "L2",
                                        in_rad = 0.3,
                                        out_rad = 2.5,
                                        apertshap = "Ring")
            pipeline.add_module(test)
            pipeline.run_module("aligner")
            
            if more_stars:
                for star in stars:
                    test = IFS_RefStarAlignment(name_in="aligner"+star, 
                                                sci_in_tag="centered", 
                                                ref_in_tags="centered_"+star, 
                                                fit_out_tag_suff="opt",
                                                qual_method = "L2",
                                                in_rad = 0.3,
                                                out_rad = 2.5,
                                                apertshap = "Ring")
                    pipeline.add_module(test)
                    pipeline.run_module("aligner"+star)
            
            # =============================================================================
            # Residual
            # =============================================================================


            resid_calc = IFS_ClassicalRefSubstraction(name_in='residual',
                                                      image_in_tags=['centered','centered_ref_opt'],
                                                      image_out_tag='Residual')

            pipeline.add_module(resid_calc)
            pipeline.run_module("residual")
            
            if more_stars:
                for star in stars:
                    resid_calc = IFS_ClassicalRefSubstraction(name_in='residual'+star,
                                                              image_in_tags=['centered','centered_'+star+'_opt'],
                                                              image_out_tag='Residual_'+star)

                    pipeline.add_module(resid_calc)
                    pipeline.run_module("residual"+star)
            
            


    # TODO: properly output results! and check if header info needs to be saved as well

            # 7.) Get the data from the Pynpoint database
            result_dict = dict()
            residuals = []
            residuals.append(pipeline.get_data("Residual"))
            if more_stars:
               for star in stars:
                   residuals.append(pipeline.get_data("Residual_"+star))
            
            for idx, tmp_algo_name in enumerate(self.get_method_keys()):
                result_dict[tmp_algo_name] = residuals[idx]
            
            # # 7.) Get the data from the Pynpoint database
            # result_dict = dict()

            # residuals = pipeline.get_data("Residual")
            # result_dict["Residual"] = residuals

            # Delete the temporary database
            shutil.rmtree(pynpoint_dir)

            # Enable print messages again
            sys.stdout = sys.__stdout__
            print("All PynPoint Modules have run successfully. Finishing up...")
            
            if self.publish_pipeline:
                return result_dict, pipeline
            
        return result_dict, None
    
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


class JWSTPCASubtractionPynPoint_unopt(DataReductionInterface):
    """
    The JWSTSimpleSubtractionPynPoint is a wrapper around the Simple Subtraction
    implemented in `Pynpoint_ifs`__.
    """

    def __init__(
            self,
            num_pcas: List[int],
            scratch_dir: Path,
            image_ref: np.ndarray,
            access_pipeline: Optional[bool] = False):
        """
        Constructor of the class.

        Args:
            scratch_dir: A directory to store the Pynpoint database. Any
                Pynpoint database created during the computation will be deleted
                afterwards.
            psf_list: List of psf template directories. If not given only
                the psf_template inputted in __call__ will be used.
        """

        self.num_pcas = num_pcas
        self.scratch_dir = scratch_dir
        self.publish_pipeline = access_pipeline
        self.image_ref = image_ref

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
            stack_dir: str,
            psf_template: str,
            parang_rad,
            exp_id: str
    ) -> Dict[str, np.ndarray]:


        pynpoint_dir = "tmp_pynpoint_" + exp_id
        pynpoint_dir = Path(self.scratch_dir + "/" + pynpoint_dir)

        if not pynpoint_dir.is_dir():
            pynpoint_dir.mkdir()

        out_file = h5py.File(
            pynpoint_dir / "PynPoint_database.hdf5",
            mode='w')

        out_file.create_dataset("data_with_planet", data=stack_with_fake_planet)

        tags = []
        for i in range(self.image_ref.shape[0]):
            out_file.create_dataset('pca_ref' + str(i), data=self.image_ref[i, :, :])
            tags.append('pca_ref' + str(i))

        # add header information to data_set?
        out_file.close()



        # 6.) Create PynPoint Pipeline and run PCA
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Disable all print messages from pynpoint
            sys.stdout = open(os.devnull, 'w')

            pipeline = Pypeline(working_place_in=str(pynpoint_dir),
                                input_place_in=str(pynpoint_dir),
                                output_place_in=str(pynpoint_dir))

            self.set_config_file()

            pca_subtraction = IFS_PCASubtraction(name_in='pca_subtraction',
                                                 image_in_tag='data_with_planet',
                                                 image_out_tag='residuals',
                                                 image_ref_in_tags=tags,
                                                 model_out_tag='pca_model',
                                                 n_pc=self.num_pcas,
                                                 pca_out_tag='Normed_PCA_components')

            pipeline.add_module(pca_subtraction)
            pipeline.run_module("pca_subtraction")

        result_dict = dict()
        residuals = pipeline.get_data("residuals")

        for idx, tmp_algo_name in enumerate(self.get_method_keys()):
            result_dict[tmp_algo_name] = residuals[idx]

        # Delete the temporary database
        shutil.rmtree(pynpoint_dir)

        # Enable print messages again
        sys.stdout = sys.__stdout__

        return result_dict, None

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