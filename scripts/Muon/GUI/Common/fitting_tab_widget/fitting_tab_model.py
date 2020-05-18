# Mantid Repository : https://github.com/mantidproject/mantid
#
# Copyright &copy; 2018 ISIS Rutherford Appleton Laboratory UKRI,
#   NScD Oak Ridge National Laboratory, European Spallation Source,
#   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
# SPDX - License - Identifier: GPL - 3.0 +
from typing import NamedTuple, List

from Muon.GUI.Common.contexts.fitting_context import FitInformation
from Muon.GUI.Common.utilities.algorithm_utils import run_Fit, run_simultaneous_Fit, run_CalculateMuonAsymmetry
from Muon.GUI.Common.ADSHandler.workspace_naming import *
from Muon.GUI.Common.ADSHandler.muon_workspace_wrapper import MuonWorkspaceWrapper
from mantid.simpleapi import (RenameWorkspace, ConvertFitFunctionForMuonTFAsymmetry, CalculateMuonAsymmetry,
                              CopyLogs, EvaluateFunction)
from mantid.api import AnalysisDataService
import mantid

MUON_ANALYSIS_SUFFIX = ' MA'
FREQUENCY_DOMAIN_ANALYSIS_SUFFIX = ' FD'
MUON_ANALYSIS_GUESS_WS = '__muon_analysis_fitting_guess'
FREQUENCY_DOMAIN_ANALYSIS_GUESS_WS = '__frequency_domain_analysis_fitting_guess'


class FitPlotInformation(NamedTuple):
    input_workspaces: List[str]
    fit: FitInformation


class FittingTabModel(object):
    def __init__(self, context):
        self.context = context
        self.function_name = ''
        self._grppair_index = {}

        # fitting options from view
        self.fit_function = None
        self.startX = 0
        self.endX = 0
        self.minimiser = ''
        self.evaluation_type = ''
        self.fit_to_raw = True
        self.fit_type = "Single"
        self.fit_by = "None"
        self.global_parameters = []
        self.tf_asymmetry_mode = False
        self.ws_fit_function_map = {}

    @property
    def function_name(self):
        return self._function_name

    @function_name.setter
    def function_name(self, value):
        if value:
            self._function_name = ' ' + value
        else:
            self._function_name = ''

    @property
    def stored_fit_functions(self):
        return list(self.ws_fit_function_map.values())

    def get_function_name(self, function):
        if function is None:
            return ''

        if function.getNumberDomains() > 1:
            function_temp = function.getFunction(0)
        else:
            function_temp = function

        try:
            function_string_list = []
            for i in range(function_temp.nFunctions()):
                function_string_list.append(function_temp.getFunction(i).name())
            if len(function_string_list) > 3:
                function_string_list = function_string_list[:3]
                function_string_list.append('...')
            function_string = ','.join(function_string_list)
            return function_string
        except AttributeError:
            return function_temp.name()

    # single fitting
    def evaluate_single_fit(self, workspace):

        if self.fit_type == "Single":
            if self.tf_asymmetry_mode:
                params = self.get_parameters_for_single_tf_fit(workspace[0])
                function_object, output_status, output_chi_squared = self.do_single_tf_fit(params)
            else:
                params = self.get_parameters_for_single_fit(workspace[0])
                function_object, output_status, output_chi_squared = self.do_single_fit(params)
        else:  # single simultaneous fit
            if self.tf_asymmetry_mode:
                params = self.get_parameters_for_simultaneous_tf_fit(workspace)
                function_object, output_status, output_chi_squared = self.do_simultaneous_tf_fit(params,
                                                                                                 self.global_parameters)
            else:
                params = self.get_parameters_for_simultaneous_fit(workspace)
                function_object, output_status, output_chi_squared = self.do_simultaneous_fit(params,
                                                                                              self.global_parameters)
        return function_object, output_status, output_chi_squared

    def do_single_fit(self, parameter_dict):
        output_workspace, fitting_parameters_table, function_object, output_status, output_chi_squared, covariance_matrix = \
            self.do_single_fit_and_return_workspace_parameters_and_fit_function(parameter_dict)

        self._handle_single_fit_results(parameter_dict['InputWorkspace'], function_object, fitting_parameters_table,
                                        output_workspace, covariance_matrix)

        return function_object, output_status, output_chi_squared

    def do_single_tf_fit(self, parameter_dict):
        alg = mantid.AlgorithmManager.create("CalculateMuonAsymmetry")
        output_workspace, fitting_parameters_table, function_object, output_status, output_chi_squared, covariance_matrix = \
            run_CalculateMuonAsymmetry(parameter_dict, alg)
        CopyLogs(InputWorkspace=parameter_dict['ReNormalizedWorkspaceList'], OutputWorkspace=output_workspace,
                 StoreInADS=False)
        self._handle_single_fit_results(parameter_dict['ReNormalizedWorkspaceList'], function_object,
                                        fitting_parameters_table, output_workspace, covariance_matrix)

        return function_object, output_status, output_chi_squared

    def do_single_fit_and_return_workspace_parameters_and_fit_function(
            self, parameters_dict):
        alg = mantid.AlgorithmManager.create("Fit")
        output_workspace, output_parameters, function_object, output_status, output_chi, covariance_matrix = run_Fit(
            parameters_dict, alg)
        CopyLogs(InputWorkspace=parameters_dict['InputWorkspace'], OutputWorkspace=output_workspace, StoreInADS=False)
        return output_workspace, output_parameters, function_object, output_status, output_chi, covariance_matrix

    def _handle_single_fit_results(self, input_workspace, fit_function, fitting_parameters_table, output_workspace,
                                   covariance_matrix):
        workspace_name, workspace_directory = create_fitted_workspace_name(input_workspace, self.function_name)
        table_name, table_directory = create_parameter_table_name(input_workspace, self.function_name)

        self.add_workspace_to_ADS(output_workspace, workspace_name, workspace_directory)
        self.add_workspace_to_ADS(covariance_matrix, workspace_name + '_CovarianceMatrix', table_directory)
        wrapped_parameter_workspace = self.add_workspace_to_ADS(fitting_parameters_table, table_name, table_directory)

        self.add_fit_to_context(wrapped_parameter_workspace,
                                fit_function,
                                input_workspace, [workspace_name])

    def do_simultaneous_fit(self, parameter_dict, global_parameters):
        output_workspace, fitting_parameters_table, function_object, output_status, output_chi_squared, covariance_matrix = \
            self.do_simultaneous_fit_and_return_workspace_parameters_and_fit_function(parameter_dict)
        self._handle_simultaneous_fit_results(parameter_dict['InputWorkspace'], function_object,
                                              fitting_parameters_table, output_workspace, global_parameters,
                                              covariance_matrix)

        return function_object, output_status, output_chi_squared

    def do_simultaneous_fit_and_return_workspace_parameters_and_fit_function(
            self, parameters_dict):
        alg = mantid.AlgorithmManager.create("Fit")

        output_workspace, output_parameters, function_object, output_status, output_chi, covariance_matrix \
            = run_simultaneous_Fit(parameters_dict, alg)
        if len(parameters_dict['InputWorkspace']) > 1:
            for input_workspace, output in zip(parameters_dict['InputWorkspace'],
                                               mantid.api.AnalysisDataService.retrieve(output_workspace).getNames()):
                CopyLogs(InputWorkspace=input_workspace, OutputWorkspace=output, StoreInADS=False)
        else:
            CopyLogs(InputWorkspace=parameters_dict['InputWorkspace'][0], OutputWorkspace=output_workspace,
                     StoreInADS=False)
        return output_workspace, output_parameters, function_object, output_status, output_chi, covariance_matrix

    def do_simultaneous_tf_fit(self, parameter_dict, global_parameters):
        alg = mantid.AlgorithmManager.create("CalculateMuonAsymmetry")
        output_workspace, fitting_parameters_table, function_object, output_status, output_chi_squared, covariance_matrix = \
            run_CalculateMuonAsymmetry(parameter_dict, alg)
        if len(parameter_dict['ReNormalizedWorkspaceList']) > 1:
            for input_workspace, output in zip(parameter_dict['ReNormalizedWorkspaceList'],
                                               mantid.api.AnalysisDataService.retrieve(output_workspace).getNames()):
                CopyLogs(InputWorkspace=input_workspace, OutputWorkspace=output, StoreInADS=False)
        else:
            CopyLogs(InputWorkspace=parameter_dict['ReNormalizedWorkspaceList'][0], OutputWorkspace=output_workspace,
                     StoreInADS=False)

        self._handle_simultaneous_fit_results(parameter_dict['ReNormalizedWorkspaceList'], function_object,
                                              fitting_parameters_table, output_workspace, global_parameters,
                                              covariance_matrix)

        return function_object, output_status, output_chi_squared

    # sequential fitting
    def evaluate_sequential_fit(self, workspaces, use_initial_values):
        # workspaces are stored as list of list [[Fit1 workspaces], [Fit2 workspaces], [Fit3 workspaces]]
        if self.fit_type == "Single":
            # flatten the workspace list
            workspace_list = [workspace for fit_workspaces in workspaces for workspace in fit_workspaces]
            if self.tf_asymmetry_mode:
                function_object, output_status, output_chi_squared = self.do_sequential_tf_fit(workspace_list,
                                                                                               use_initial_values)
            else:
                function_object, output_status, output_chi_squared = self.do_sequential_fit(workspace_list,
                                                                                            use_initial_values)
        else:
            # in a simultaneous-sequential fit, each fit corresponds to a list of workspaces
            if self.tf_asymmetry_mode:
                function_object, output_status, output_chi_squared = \
                    self.do_sequential_simultaneous_tf_fit(workspaces, use_initial_values)
            else:
                function_object, output_status, output_chi_squared = \
                    self.do_sequential_simultaneous_fit(workspaces, use_initial_values)

        return function_object, output_status, output_chi_squared

    def do_sequential_fit(self, workspace_list, use_initial_values=False):
        function_object_list = []
        output_status_list = []
        output_chi_squared_list = []

        for i, input_workspace in enumerate(workspace_list):
            params = self.get_parameters_for_single_fit(input_workspace)

            if not use_initial_values and i >= 1:
                previous_values = self.get_fit_function_parameter_values(function_object_list[i - 1])
                self.set_fit_function_parameter_values(params['Function'],
                                                       previous_values)

            function_object, output_status, output_chi_squared = self.do_single_fit(params)

            function_object_list.append(function_object)
            output_status_list.append(output_status)
            output_chi_squared_list.append(output_chi_squared)

        return function_object_list, output_status_list, output_chi_squared_list

    def do_sequential_simultaneous_fit(self, workspaces, use_initial_values=False):
        function_object_list = []
        output_status_list = []
        output_chi_squared_list = []

        # workspaces defines a list of lists [[ws1,ws2],[ws1,ws2]...]
        for i, workspace_list in enumerate(workspaces):
            params = self.get_parameters_for_simultaneous_fit(workspace_list)

            if not use_initial_values and i >= 1:
                previous_values = self.get_fit_function_parameter_values(function_object_list[i - 1])
                self.set_fit_function_parameter_values(params['Function'],
                                                       previous_values)

            function_object, output_status, output_chi_squared = self.do_simultaneous_fit(params,
                                                                                          self.global_parameters)
            function_object_list.append(function_object)
            output_status_list.append(output_status)
            output_chi_squared_list.append(output_chi_squared)

        return function_object_list, output_status_list, output_chi_squared_list

    def do_sequential_tf_fit(self, workspace_list, use_initial_values=False):
        function_object_list = []
        output_status_list = []
        output_chi_squared_list = []

        for i, input_workspace in enumerate(workspace_list):
            params = self.get_parameters_for_single_tf_fit(input_workspace)

            if not use_initial_values and i >= 1:
                previous_values = self.get_fit_function_parameter_values(function_object_list[i - 1])
                self.set_fit_function_parameter_values(params['InputFunction'],
                                                       previous_values)

            function_object, output_status, output_chi_squared = self.do_single_tf_fit(params)

            function_object_list.append(function_object)
            output_status_list.append(output_status)
            output_chi_squared_list.append(output_chi_squared)

        return function_object_list, output_status_list, output_chi_squared_list

    def do_sequential_simultaneous_tf_fit(self, workspaces, use_initial_values=False):
        function_object_list = []
        output_status_list = []
        output_chi_squared_list = []

        for i, workspace_list in enumerate(workspaces):
            params = self.get_parameters_for_simultaneous_tf_fit(workspace_list)

            if not use_initial_values and i >= 1:
                previous_values = self.get_fit_function_parameter_values(function_object_list[i - 1])
                self.set_fit_function_parameter_values(params['InputFunction'],
                                                       previous_values)

            function_object, output_status, output_chi_squared = self.do_simultaneous_tf_fit(params,
                                                                                             self.global_parameters)

            function_object_list.append(function_object)
            output_status_list.append(output_status)
            output_chi_squared_list.append(output_chi_squared)

        return function_object_list, output_status_list, output_chi_squared_list

    def _handle_simultaneous_fit_results(self, input_workspace_list, fit_function, fitting_parameters_table,
                                         output_workspace, global_parameters, covariance_matrix):
        if len(input_workspace_list) > 1:
            table_name, table_directory = create_parameter_table_name(input_workspace_list[0] + '+ ...',
                                                                      self.function_name)
            workspace_name, workspace_directory = create_multi_domain_fitted_workspace_name(
                input_workspace_list[0], self.function_name)
            self.add_workspace_to_ADS(output_workspace, workspace_name, '')

            workspace_name = self.rename_members_of_fitted_workspace_group(workspace_name,
                                                                           input_workspace_list,
                                                                           fit_function)
            self.add_workspace_to_ADS(covariance_matrix, workspace_name[0] + '_CovarianceMatrix', table_directory)
        else:
            table_name, table_directory = create_parameter_table_name(input_workspace_list[0], self.function_name)
            workspace_name, workspace_directory = create_fitted_workspace_name(input_workspace_list[0],
                                                                               self.function_name)
            self.add_workspace_to_ADS(output_workspace, workspace_name, workspace_directory)
            self.add_workspace_to_ADS(covariance_matrix, workspace_name + '_CovarianceMatrix', table_directory)
            workspace_name = [workspace_name]

        wrapped_parameter_workspace = self.add_workspace_to_ADS(fitting_parameters_table, table_name,
                                                                table_directory)

        self.add_fit_to_context(wrapped_parameter_workspace,
                                fit_function,
                                input_workspace_list,
                                workspace_name,
                                global_parameters)

    # workspace operations
    def rename_members_of_fitted_workspace_group(self, group_workspace, inputworkspace_list, function):
        self.context.ads_observer.observeRename(False)
        output_workspace_list = []
        for index, workspace_name in enumerate(AnalysisDataService.retrieve(group_workspace).getNames()):
            new_name, _ = create_fitted_workspace_name(inputworkspace_list[index], self.function_name)

            new_name += '; Simultaneous'
            output_workspace_list.append(new_name)
            RenameWorkspace(InputWorkspace=workspace_name,
                            OutputWorkspace=new_name)

        self.context.ads_observer.observeRename(True)

        return output_workspace_list

    def add_workspace_to_ADS(self, workspace, name, directory):
        self.context.ads_observer.observeRename(False)
        workspace_wrapper = MuonWorkspaceWrapper(workspace)
        workspace_wrapper.show(directory + name)
        self.context.ads_observer.observeRename(True)
        return workspace_wrapper

    def add_fit_to_context(self, parameter_workspace, function,
                           input_workspace, output_workspace_name, global_parameters=None):
        self.context.fitting_context.add_fit_from_values(
            parameter_workspace, self.function_name,
            input_workspace, output_workspace_name, global_parameters)

    def change_plot_guess(self, plot_guess, parameter_dict):
        try:
            fit_function = parameter_dict['Function']
            data_ws_name = parameter_dict['InputWorkspace']
        except KeyError:
            return
        if self.context.workspace_suffix == MUON_ANALYSIS_SUFFIX:
            guess_ws_name = MUON_ANALYSIS_GUESS_WS
        elif self.context.workspace_suffix == FREQUENCY_DOMAIN_ANALYSIS_SUFFIX:
            guess_ws_name = FREQUENCY_DOMAIN_ANALYSIS_GUESS_WS
        else:
            guess_ws_name = '__unknown_interface_fitting_guess'

        # Handle case of function removed
        if fit_function is None and plot_guess:
            self.context.fitting_context.notify_plot_guess_changed(plot_guess, None)
        elif fit_function is None or data_ws_name == '':
            return
        else:
            # evaluate the current function on the workspace
            if plot_guess:
                try:
                    EvaluateFunction(InputWorkspace=data_ws_name,
                                     Function=fit_function,
                                     StartX=parameter_dict['StartX'],
                                     EndX=parameter_dict['EndX'],
                                     OutputWorkspace=guess_ws_name)
                except RuntimeError:
                    mantid.logger.error('Could not evaluate the function.')
                    return

            if AnalysisDataService.doesExist(guess_ws_name):
                self.context.fitting_context.notify_plot_guess_changed(plot_guess, guess_ws_name)

    def update_plot_guess(self, fit_function, workspace_name):

        if self.context.workspace_suffix == MUON_ANALYSIS_SUFFIX:
            guess_ws_name = MUON_ANALYSIS_GUESS_WS
        elif self.context.workspace_suffix == FREQUENCY_DOMAIN_ANALYSIS_SUFFIX:
            guess_ws_name = FREQUENCY_DOMAIN_ANALYSIS_GUESS_WS
        else:
            guess_ws_name = '__unknown_interface_fitting_guess'
        try:
            EvaluateFunction(InputWorkspace=workspace_name,
                             Function=fit_function,
                             StartX=self.startX,
                             EndX=self.endX,
                             OutputWorkspace=guess_ws_name)
        except RuntimeError:
            mantid.logger.error('Could not evaluate the function.')
            return
        if AnalysisDataService.doesExist(guess_ws_name):
            self.context.fitting_context.notify_plot_guess_changed(True, guess_ws_name)

    # update model information
    def update_stored_fit_function(self, fit_function):
        self.fit_function = fit_function

    def update_model_fit_options(self, fit_type,
                                 fit_specifier,
                                 fit_function,
                                 start_x,
                                 end_x,
                                 minimiser,
                                 evaluation_type,
                                 fit_to_raw,
                                 global_parameters,
                                 tf_asymmetry_mode):

        self.fit_type = fit_type
        self.fit_by = fit_specifier
        self.fit_function = fit_function
        self.startX = start_x
        self.endX = end_x
        self.minimiser = minimiser
        self.evaluation_type = evaluation_type
        self.fit_to_raw = fit_to_raw
        self.global_parameters = global_parameters
        self.tf_asymmetry_mode = tf_asymmetry_mode

        self.create_ws_fit_function_map()

    def create_ws_fit_function_map(self):
        if self.fit_function is None:
            return

        self.ws_fit_function_map.clear()
        equiv_workspace_names = self.create_hashable_fit_workspace_names()
        for workspace in equiv_workspace_names:
            self.ws_fit_function_map[workspace] = self.fit_function.clone()

        if self.tf_asymmetry_mode:
            self.update_fit_function_map_with_tf_asym_data()

    # This function creates a workspace string from a workspace_list taken from the fit input workspaces.
    # This string is then used as a key in the ws_fit_function_map.
    def create_hashable_fit_workspace_names(self):
        runs, group_and_pairs = self.get_runs_groups_and_pairs_for_fits()
        list_of_workspace_lists = []
        for run, group_and_pair in zip(runs, group_and_pairs):
            separated_runs, separated_groups_and_pairs = \
                self.get_separated_runs_and_group_and_pairs(run, group_and_pair)
            list_of_workspace_lists += [self.get_fit_workspace_names_from_groups_and_runs(separated_runs,
                                                                                          separated_groups_and_pairs)]
        equiv_workspace_list = []
        for workspace_list in list_of_workspace_lists:
            equiv_workspace_list += [self.create_equivalent_workspace_name(workspace_list)]

        return equiv_workspace_list

    def update_ws_fit_function_parameters(self, workspace, params):
        equiv_workspace_name = self.create_equivalent_workspace_name(workspace)
        try:
            fit_function = self.ws_fit_function_map[equiv_workspace_name]
        except KeyError:
            return
        self.set_fit_function_parameter_values(fit_function, params)

    def update_fit_function_map_with_tf_asym_data(self):
        for workspaces, fit_function in self.ws_fit_function_map.items():
            workspace_list = self.get_separated_workspaces(workspaces)
            self.update_tf_fit_function(workspace_list, fit_function)

    def update_tf_fit_function(self, workspaces, fit_function):
        if self.fit_type == "Single":
            tf_asymmetry_parameters = self.get_params_for_single_tf_function_calculation(workspaces[0], fit_function)
        else:
            tf_asymmetry_parameters = self.get_params_for_multi_tf_function_calculation(workspaces, fit_function)

        fit_function = self.calculate_tf_function(tf_asymmetry_parameters)

    def get_params_for_single_tf_function_calculation(self, workspace, fit_function):
        return {
            'InputFunction': fit_function,
            'ReNormalizedWorkspaceList': workspace,
            'UnNormalizedWorkspaceList': self.context.group_pair_context.get_unormalisised_workspace_list(
                [workspace])[0],
            'OutputFitWorkspace': "fit",
            'StartX': self.startX,
            'EndX': self.endX,
            'Minimizer': self.minimiser
        }

    def get_params_for_multi_tf_function_calculation(self, workspace_list, fit_function):
        return {
            'InputFunction': fit_function,
            'ReNormalizedWorkspaceList': workspace_list,
            'UnNormalizedWorkspaceList': self.context.group_pair_context.get_unormalisised_workspace_list(
                workspace_list),
            'OutputFitWorkspace': "fit",
            'StartX': self.startX,
            'EndX': self.endX,
            'Minimizer': self.minimiser
        }

    def clear_fit_information(self):
        self.fit_function = None
        self.function_name = ''

    def freq_type(self):
        if self.context._frequency_context is not None:
            freq = self.context._frequency_context.plot_type
        else:
            freq = 'None'
        return freq

    def get_ws_fit_function(self, workspaces):
        equiv_workspace_name = self.create_equivalent_workspace_name(workspaces)
        try:
            return self.ws_fit_function_map[equiv_workspace_name]
        except KeyError:
            pass

    def get_parameters_for_single_fit(self, workspace):
        params = self._get_shared_parameters()
        params['InputWorkspace'] = workspace
        params['Function'] = self.get_ws_fit_function([workspace])
        params['StartX'] = self.startX
        params['EndX'] = self.endX

        return params

    def get_parameters_for_simultaneous_fit(self, workspaces):
        params = self._get_shared_parameters()
        params['InputWorkspace'] = workspaces
        params['Function'] = self.get_ws_fit_function(workspaces)
        params['StartX'] = [self.startX] * len(workspaces)
        params['EndX'] = [self.endX] * len(workspaces)

        return params

    def _get_shared_parameters(self):
        """
        :return: The set of attributes common to all fit types
        """
        return {
            'Minimizer': self.minimiser,
            'EvaluationType': self.evaluation_type
        }

    def get_parameters_for_single_tf_fit(self, workspace):
        # workspace is the name of the input workspace
        fit_workspace_name, _ = create_fitted_workspace_name(workspace, self.function_name)
        return {
            'InputFunction': self.get_ws_fit_function([workspace]),
            'ReNormalizedWorkspaceList': workspace,
            'UnNormalizedWorkspaceList': self.context.group_pair_context.get_unormalisised_workspace_list([workspace])[
                0],
            'OutputFitWorkspace': fit_workspace_name,
            'StartX': self.startX,
            'EndX': self.endX,
            'Minimizer': self.minimiser
        }

    def get_parameters_for_simultaneous_tf_fit(self, workspaces):
        # workspaces is a list containing the N workspaces which form the simultaneous fit
        # creates a workspace name based on the first entry in the workspace list
        fit_workspaces, _ = create_multi_domain_fitted_workspace_name(workspaces[0], self.function_name)
        return {
            'InputFunction': self.get_ws_fit_function(workspaces),
            'ReNormalizedWorkspaceList': workspaces,
            'UnNormalizedWorkspaceList': self.context.group_pair_context.get_unormalisised_workspace_list(workspaces),
            'OutputFitWorkspace': fit_workspaces,
            'StartX': self.startX,
            'EndX': self.endX,
            'Minimizer': self.minimiser
        }

    def get_parameters_for_tf_single_fit_calculation(self, workspace):
        fit_workspace, _ = create_fitted_workspace_name(workspace, self.function_name)
        return {
            'InputFunction': self.fit_function,
            'ReNormalizedWorkspaceList': workspace,
            'UnNormalizedWorkspaceList': self.context.group_pair_context.get_unormalisised_workspace_list(
                [workspace])[0],
            'OutputFitWorkspace': fit_workspace,
            'StartX': self.startX,
            'EndX': self.endX,
            'Minimizer': self.minimiser
        }

    # get workspace information
    def get_selected_workspace_list(self):
        selected_workspaces = []
        selected_groups_and_pairs = self._get_selected_groups_and_pairs()
        for grp_and_pair in selected_groups_and_pairs:
            selected_workspaces += self.context.get_names_of_workspaces_to_fit(
                runs='All',
                group_and_pair=grp_and_pair,
                phasequad=False,
                rebin=not self.fit_to_raw, freq=self.freq_type())
        return selected_workspaces

    def get_runs_groups_and_pairs_for_fits(self):
        selected_workspaces = self.get_selected_workspace_list()
        runs = []
        groups_and_pairs = []

        if self.fit_type == "Single":
            for workspace in selected_workspaces:
                runs += [get_run_number_from_workspace_name(workspace, self.context.data_context.instrument)]
                groups_and_pairs += [get_group_or_pair_from_name(workspace)]
            run_groups_and_pairs = sorted(zip(runs, groups_and_pairs))
            groups_and_pairs = [grp_pair for _, grp_pair in run_groups_and_pairs]
            runs = [run for run, _ in run_groups_and_pairs]
        else:
            fit_workspaces = {}
            if self.fit_by == "Run":
                for workspace in selected_workspaces:
                    run = get_run_number_from_workspace_name(workspace, self.context.data_context.instrument)
                    if run not in fit_workspaces:
                        fit_workspaces[run] = get_group_or_pair_from_name(workspace)
                    else:
                        fit_workspaces[run] += ";" + get_group_or_pair_from_name(workspace)
                runs = list(fit_workspaces.keys())
                runs.sort()
                groups_and_pairs = list(fit_workspaces.values())
            elif self.fit_by == "Group/Pair":
                for workspace in selected_workspaces:
                    group_or_pair = get_group_or_pair_from_name(workspace)
                    run = get_run_number_from_workspace_name(workspace, self.context.data_context.instrument)
                    if group_or_pair not in fit_workspaces:
                        fit_workspaces[group_or_pair] = run
                    else:
                        fit_workspaces[group_or_pair] += ";" + run
                runs = list(fit_workspaces.values())
                groups_and_pairs = list(fit_workspaces.keys())
            else:
                return

        return runs, groups_and_pairs

    def get_fit_workspace_names_from_groups_and_runs(self, runs, groups_and_pairs):
        workspace_names = []
        for run in runs:
            for group_or_pair in groups_and_pairs:
                if group_or_pair in self.context.group_pair_context.selected_pairs:
                    workspace_names += [get_pair_asymmetry_name(self.context, group_or_pair, run, not self.fit_to_raw)]
                else:
                    workspace_names += [get_group_asymmetry_name(self.context, group_or_pair, run, not self.fit_to_raw)]

        return workspace_names

    def workspace_list_sorter(self, workspace_name):
        run_number = get_run_number_from_workspace_name(workspace_name, self.context.data_context.instrument)
        grp_pair_number = self._transform_grp_or_pair_to_float(workspace_name)
        return int(run_number) + grp_pair_number

    # Converts the workspace group or pair name to a float which is used in sorting the workspace list
    # If there is only 1 group or pair selected this function returns 0 as the workspace list will be sorted by runs
    def _transform_grp_or_pair_to_float(self, workspace_name):
        grp_or_pair_name = get_group_or_pair_from_name(workspace_name)
        if grp_or_pair_name not in self._grppair_index:
            self._grppair_index[grp_or_pair_name] = len(self._grppair_index)

        grp_pair_values = list(self._grppair_index.values())
        if len(self._grppair_index) > 1:
            return ((self._grppair_index[grp_or_pair_name] - grp_pair_values[0])
                    / (grp_pair_values[-1] - grp_pair_values[0])) * 0.99
        else:
            return 0

    def _get_selected_groups_and_pairs(self):
        return self.context.group_pair_context.selected_groups + self.context.group_pair_context.selected_pairs

    @staticmethod
    def get_fit_function_parameter_values(fit_function):
        number_of_parameters = fit_function.nParams()
        parameters = [fit_function.parameterName(i) for i in range(number_of_parameters)]
        parameter_values = [fit_function.getParameterValue(parameters[i]) for i in
                            range(number_of_parameters)]
        return parameter_values

    @staticmethod
    def set_fit_function_parameter_values(fit_function, vals):
        number_of_parameters = fit_function.nParams()
        parameters = [fit_function.parameterName(i) for i in range(number_of_parameters)]
        for i in range(number_of_parameters):
            fit_function.setParameter(parameters[i], vals[i])

    @staticmethod
    def create_equivalent_workspace_name(workspace_list):
        eq_name = workspace_list[0]
        for i in range(1, len(workspace_list)):
            eq_name += '-' + workspace_list[i]

        return eq_name

    @staticmethod
    def convert_to_tf_function(algorithm_parameters):
        return ConvertFitFunctionForMuonTFAsymmetry(StoreInADS=False, **algorithm_parameters)

    @staticmethod
    def calculate_tf_function(algorithm_parameters):
        return CalculateMuonAsymmetry(StoreInADS=False, **algorithm_parameters)

    @staticmethod
    def get_separated_runs_and_group_and_pairs(runs, group_and_pairs):
        separated_runs = runs.split(';')
        separated_group_and_pairs = group_and_pairs.split(';')
        return separated_runs, separated_group_and_pairs

    @staticmethod
    def get_separated_workspaces(workspace_names):
        separated_workspaces = workspace_names.split('-')
        return separated_workspaces
