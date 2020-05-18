# Mantid Repository : https://github.com/mantidproject/mantid
#
# Copyright &copy; 2019 ISIS Rutherford Appleton Laboratory UKRI,
#   NScD Oak Ridge National Laboratory, European Spallation Source,
#   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
# SPDX - License - Identifier: GPL - 3.0 +
import unittest
from Muon.GUI.Common.fitting_tab_widget.fitting_tab_model import FittingTabModel
from Muon.GUI.Common.test_helpers.context_setup import setup_context
from mantid.api import FunctionFactory, AnalysisDataService
from mantid.simpleapi import CreateWorkspace
from unittest import mock
from mantidqt.utils.qt.testing import start_qapplication


@start_qapplication
class FittingTabModelTest(unittest.TestCase):

    def setUp(self):
        self.model = FittingTabModel(setup_context())
        self.model.context.ads_observer.unsubscribe()

    def test_convert_function_string_into_dict(self):
        trial_function = FunctionFactory.createInitialized('name=GausOsc,A=0.2,Sigma=0.2,Frequency=0.1,Phi=0')

        name = self.model.get_function_name(trial_function)

        self.assertEqual(name, 'GausOsc')

    def test_do_single_fit_and_return_functions_correctly(self):
        x_data = range(0, 100)
        y_data = [5 + x * x for x in x_data]
        workspace = CreateWorkspace(x_data, y_data)
        trial_function = FunctionFactory.createInitialized('name = Quadratic, A0 = 0, A1 = 0, A2 = 0')
        parameter_dict = {'Function': trial_function, 'InputWorkspace': workspace, 'Minimizer': 'Levenberg-Marquardt',
                          'StartX': 0.0, 'EndX': 100.0, 'EvaluationType': 'CentrePoint'}

        output_workspace, parameter_table_name, fitting_function, fit_status, fit_chi_squared, covariance_matrix = self.model.do_single_fit_and_return_workspace_parameters_and_fit_function(
            parameter_dict)

        parameter_table = AnalysisDataService.retrieve(parameter_table_name)

        self.assertAlmostEqual(parameter_table.row(0)['Value'], 5.0)
        self.assertAlmostEqual(parameter_table.row(1)['Value'], 0.0)
        self.assertAlmostEqual(parameter_table.row(2)['Value'], 1.0)
        self.assertEqual(fit_status, 'success')
        self.assertAlmostEqual(fit_chi_squared, 0.0)

    def test_add_workspace_to_ADS_adds_workspace_to_ads_in_correct_group_structure(self):
        workspace = CreateWorkspace([0, 0], [0, 0])
        workspace_name = 'test_workspace_name'
        workspace_directory = 'root/'

        self.model.add_workspace_to_ADS(workspace, workspace_name, workspace_directory)

        self.assertTrue(AnalysisDataService.doesExist(workspace_name))
        self.assertTrue(AnalysisDataService.doesExist('root'))

    def test_do_sequential_fit_correctly_delegates_to_do_single_fit(self):
        trial_function = FunctionFactory.createInitialized('name = Quadratic, A0 = 0, A1 = 0, A2 = 0')
        self.model.do_single_fit = mock.MagicMock(return_value=(trial_function, 'success', 0.56))
        workspace = "MUSR1223"
        parameter_dict = {'Function': trial_function, 'InputWorkspace': workspace,
                          'Minimizer': 'Levenberg-Marquardt',
                          'StartX': 0.0, 'EndX': 100.0, 'EvaluationType': 'CentrePoint'}
        self.model.get_parameters_for_single_fit = mock.MagicMock(return_value=parameter_dict)
        self.model.do_sequential_fit([workspace] * 5)

        self.assertEqual(self.model.do_single_fit.call_count, 5)
        self.model.do_single_fit.assert_called_with(
            {'Function': mock.ANY, 'InputWorkspace': workspace, 'Minimizer': 'Levenberg-Marquardt',
             'StartX': 0.0, 'EndX': 100.0, 'EvaluationType': 'CentrePoint'})

    def test_do_simultaneous_fit_adds_single_input_workspace_to_fit_context_with_globals(self):
        trial_function = FunctionFactory.createInitialized('name = Quadratic, A0 = 0, A1 = 0, A2 = 0')
        x_data = range(0, 100)
        y_data = [5 + x * x for x in x_data]
        workspace = CreateWorkspace(x_data, y_data)
        parameter_dict = {'Function': trial_function, 'InputWorkspace': [workspace.name()],
                          'Minimizer': 'Levenberg-Marquardt',
                          'StartX': [0.0], 'EndX': [100.0], 'EvaluationType': 'CentrePoint'}
        global_parameters = ['A0']
        self.model.do_simultaneous_fit(parameter_dict, global_parameters)

        fit_context = self.model.context.fitting_context
        self.assertEqual(1, len(fit_context))
        self.assertEqual(global_parameters, fit_context.fit_list[0].parameters.global_parameters)

    def test_do_simultaneous_fit_adds_multi_input_workspace_to_fit_context(self):
        # create function
        single_func = ';name=FlatBackground,$domains=i,A0=0'
        multi_func = 'composite=MultiDomainFunction,NumDeriv=1' + single_func + single_func + ";"
        trial_function = FunctionFactory.createInitialized(multi_func)
        x_data = range(0, 100)
        y_data = [5 + x * x for x in x_data]
        workspace1 = CreateWorkspace(x_data, y_data)
        workspace2 = CreateWorkspace(x_data, y_data)
        parameter_dict = {'Function': trial_function, 'InputWorkspace': [workspace1.name(), workspace2.name()],
                          'Minimizer': 'Levenberg-Marquardt',
                          'StartX': [0.0] * 2, 'EndX': [100.0] * 2, 'EvaluationType': 'CentrePoint'}
        self.model.do_simultaneous_fit(parameter_dict, global_parameters=[])

        fit_context = self.model.context.fitting_context
        self.assertEqual(1, len(fit_context))

    def test_get_function_name_returns_correctly_for_composite_functions(self):
        function_string = 'name=FlatBackground,A0=22.5129;name=Polynomial,n=0,A0=-22.5221;name=ExpDecayOsc,A=-0.172352,Lambda=0.111109,Frequency=-0.280031,Phi=-3.03983'
        function_object = FunctionFactory.createInitialized(function_string)

        name_as_string = self.model.get_function_name(function_object)

        self.assertEqual(name_as_string, 'FlatBackground,Polynomial,ExpDecayOsc')

    def test_get_function_name_returns_correctly_for_composite_functions_multi_domain_function(self):
        function_string = 'composite=MultiDomainFunction,NumDeriv=true;(composite=CompositeFunction,NumDeriv=false,' \
                          '$domains=i;name=FlatBackground,A0=-0.800317;name=Polynomial,n=0,A0=0.791112;name=ExpDecayOsc,' \
                          'A=0.172355,Lambda=0.111114,Frequency=0.280027,Phi=-0.101717);(composite=CompositeFunction,' \
                          'NumDeriv=false,$domains=i;name=FlatBackground,A0=1.8125;name=Polynomial,n=0,A0=-1.81432;' \
                          'name=ExpDecayOsc,A=0.17304,Lambda=0.102673,Frequency=0.278695,Phi=-0.0461353);' \
                          '(composite=CompositeFunction,NumDeriv=false,$domains=i;name=FlatBackground,A0=1.045;' \
                          'name=Polynomial,n=0,A0=-1.04673;name=ExpDecayOsc,A=-0.170299,Lambda=0.118256,Frequency=0.281085,Phi=-0.155812)'
        function_object = FunctionFactory.createInitialized(function_string)

        name_as_string = self.model.get_function_name(function_object)

        self.assertEqual(name_as_string, 'FlatBackground,Polynomial,ExpDecayOsc')

    def test_get_function_name_truncates_function_with_more_than_three_composite_members(self):
        function_string = 'name=ExpDecayOsc,A=-5.87503,Lambda=0.0768409,Frequency=0.0150173,Phi=-1.15833;name=GausDecay,' \
                          'A=-1.59276,Sigma=0.361339;name=ExpDecayOsc,A=-5.87503,Lambda=0.0768409,Frequency=0.0150173,' \
                          'Phi=-1.15833;name=ExpDecayMuon,A=-0.354664,Lambda=-0.15637;name=DynamicKuboToyabe,' \
                          'BinWidth=0.050000000000000003,Asym=7.0419,Delta=0.797147,Field=606.24,Nu=2.67676e-09'
        function_object = FunctionFactory.createInitialized(function_string)

        name_as_string = self.model.get_function_name(function_object)

        self.assertEqual(name_as_string, 'ExpDecayOsc,GausDecay,ExpDecayOsc,...')

    def test_get_function_name_does_truncate_for_exactly_four_members(self):
        function_string = 'name=ExpDecayOsc,A=-5.87503,Lambda=0.0768409,Frequency=0.0150173,Phi=-1.15833;name=GausDecay,' \
                          'A=-1.59276,Sigma=0.361339;name=ExpDecayOsc,A=-5.87503,Lambda=0.0768409,Frequency=0.0150173,' \
                          'Phi=-1.15833;name=ExpDecayMuon,A=-0.354664,Lambda=-0.15637'
        function_object = FunctionFactory.createInitialized(function_string)

        name_as_string = self.model.get_function_name(function_object)

        self.assertEqual(name_as_string, 'ExpDecayOsc,GausDecay,ExpDecayOsc,...')

    def test_change_plot_guess_returns_when_given_bad_parameters(self):
        self.model.context = mock.Mock()
        self.model.change_plot_guess(True, {})

        self.assertEqual(0, self.model.context.fitting_context.notify_plot_guess_changed.call_count)

    def test_change_plot_guess_notifies_of_change_when_function_removed_but_plot_guess_true(self):
        self.model.context = mock.Mock()
        self.model.change_plot_guess(True, {'Function': None, 'InputWorkspace': 'ws'})

        self.model.context.fitting_context.notify_plot_guess_changed.assert_called_with(True, None)

    def test_change_plot_guess_evaluates_the_function(self):
        self.model.context = mock.Mock()
        with mock.patch('Muon.GUI.Common.fitting_tab_widget.fitting_tab_model.EvaluateFunction') as mock_evaluate:
            parameters = {'Function': 'func',
                          'InputWorkspace': 'ws',
                          'StartX': 0,
                          'EndX': 1}
            self.model.change_plot_guess(True, parameters)
            mock_evaluate.assert_called_with(InputWorkspace='ws',
                                             Function='func',
                                             StartX=parameters['StartX'],
                                             EndX=parameters['EndX'],
                                             OutputWorkspace='__unknown_interface_fitting_guess')

    @mock.patch('Muon.GUI.Common.fitting_tab_widget.fitting_tab_model.mantid')
    @mock.patch('Muon.GUI.Common.fitting_tab_widget.fitting_tab_model.EvaluateFunction')
    def test_change_plot_guess_writes_log_and_returns_if_function_evaluation_fails(self, mock_evaluate, mock_mantid):
        self.model.context = mock.Mock()
        mock_evaluate.side_effect = RuntimeError()
        parameters = {'Function': 'func',
                      'InputWorkspace': 'ws',
                      'StartX': 0,
                      'EndX': 1}
        self.model.change_plot_guess(True, parameters)
        mock_evaluate.assert_called_with(InputWorkspace='ws',
                                         Function='func',
                                         StartX=parameters['StartX'],
                                         EndX=parameters['EndX'],
                                         OutputWorkspace='__unknown_interface_fitting_guess')
        mock_mantid.logger.error.assert_called_with('Could not evaluate the function.')
        self.assertEqual(0, self.model.context.fitting_context.notify_plot_guess_changed.call_count)

    @mock.patch('Muon.GUI.Common.fitting_tab_widget.fitting_tab_model.AnalysisDataService')
    @mock.patch('Muon.GUI.Common.fitting_tab_widget.fitting_tab_model.EvaluateFunction')
    def test_change_plot_guess_notifies_subscribers_if_workspace_in_ads(self, mock_evaluate, mock_ads):
        self.model.context = mock.Mock()
        mock_ads.doesExist.return_value = True
        parameters = {'Function': 'func',
                      'InputWorkspace': 'ws',
                      'StartX': 0,
                      'EndX': 1}
        self.model.change_plot_guess(True, parameters)
        mock_evaluate.assert_called_with(InputWorkspace='ws',
                                         Function='func',
                                         StartX=parameters['StartX'],
                                         EndX=parameters['EndX'],
                                         OutputWorkspace='__unknown_interface_fitting_guess')
        self.assertEqual(1, self.model.context.fitting_context.notify_plot_guess_changed.call_count)
        self.model.context.fitting_context.notify_plot_guess_changed.assert_called_with(True,
                                                                                        '__unknown_interface_fitting_guess')

    def test_evaluate_single_fit_calls_correctly_for_single_fit(self):
        self.model.fit_type = "Single"
        workspace = ["MUSR:62260;bwd"]
        trial_function = FunctionFactory.createInitialized('name = Quadratic, A0 = 0, A1 = 0, A2 = 0')
        self.model.do_single_fit = mock.MagicMock(return_value=(trial_function, 'success', 0.56))
        parameter_dict = {'Test': 0}
        self.model.get_parameters_for_single_fit = mock.MagicMock(return_value=parameter_dict)

        self.model.evaluate_single_fit(workspace)

        self.model.do_single_fit.assert_called_once_with(parameter_dict)

    def test_evaluate_single_fit_calls_correct_function_for_simultaneous_fit(self):
        self.model.fit_type = "Simultaneous"
        workspace = ["MUSR:62260;bwd"]
        self.model.global_parameters = ['A0']
        trial_function = FunctionFactory.createInitialized('name = Quadratic, A0 = 0, A1 = 0, A2 = 0')
        self.model.do_simultaneous_fit = mock.MagicMock(return_value=(trial_function, 'success', 0.56))
        parameter_dict = {'Test': 0}
        self.model.get_parameters_for_simultaneous_fit = mock.MagicMock(return_value=parameter_dict)

        self.model.evaluate_single_fit(workspace)

        self.model.do_simultaneous_fit.assert_called_once_with(parameter_dict, ['A0'])

    def test_evaluate_single_fit_calls_correctly_for_single_tf_fit(self):
        self.model.fit_type = "Single"
        self.model.tf_asymmetry_mode = True
        workspace = ["MUSR:62260;bwd"]
        trial_function = FunctionFactory.createInitialized('name = Quadratic, A0 = 0, A1 = 0, A2 = 0')
        self.model.do_single_tf_fit = mock.MagicMock(return_value=(trial_function, 'success', 0.56))
        parameter_dict = {'Test': 0}
        self.model.get_parameters_for_single_tf_fit = mock.MagicMock(return_value=parameter_dict)

        self.model.evaluate_single_fit(workspace)

        self.model.do_single_tf_fit.assert_called_once_with(parameter_dict)

    def test_evaluate_single_fit_calls_correct_function_for_simultaneous_tf_fit(self):
        self.model.fit_type = "Simultaneous"
        self.model.tf_asymmetry_mode = True
        workspace = ["MUSR:62260;bwd"]
        self.model.global_parameters = ['A0']
        trial_function = FunctionFactory.createInitialized('name = Quadratic, A0 = 0, A1 = 0, A2 = 0')
        self.model.do_simultaneous_tf_fit = mock.MagicMock(return_value=(trial_function, 'success', 0.56))
        parameter_dict = {'Test': 0}
        self.model.get_parameters_for_simultaneous_tf_fit = mock.MagicMock(return_value=parameter_dict)

        self.model.evaluate_single_fit(workspace)

        self.model.do_simultaneous_tf_fit.assert_called_once_with(parameter_dict, ['A0'])

    def test_do_sequential_fit_calls_fetches_calls_single_fit_correctly(self):
        workspace_list = ["MUSR62260;bwd", "MUSR62260;fwd"]
        self.model.tf_asymmetry_mode = False
        use_initial_values = True
        self.model.do_single_fit = mock.MagicMock(return_value=("test", 'success', 0.56))

        self.model.do_sequential_fit(workspace_list, use_initial_values)

        self.assertEqual(self.model.do_single_fit.call_count, 2)

    def test_do_sequential_fit_uses_previous_values_if_requested(self):
        workspace_list = ["MUSR62260;bwd", "MUSR62260;fwd"]
        self.model.tf_asymmetry_mode = False
        use_initial_values = False
        self.model.set_fit_function_parameter_values = mock.MagicMock()
        trial_function_in = FunctionFactory.createInitialized('name = Quadratic, A0 = 0, A1 = 0, A2 = 0')
        trial_function_out = FunctionFactory.createInitialized('name = Quadratic, A0 = 5, A1 = 5, A2 = 5')
        self.model.do_single_fit = mock.MagicMock(return_value=(trial_function_out, 'success', 0.56))
        parameter_dict = {'Function': trial_function_in}
        self.model.get_parameters_for_single_fit = mock.MagicMock(return_value=parameter_dict)

        self.model.do_sequential_fit(workspace_list, use_initial_values)

        self.model.set_fit_function_parameter_values.assert_called_once_with(trial_function_in, [5, 5, 5])

    def test_create_equivalent_workspace_name_returns_expected_name(self):
        workspace_list = ["MUSR62260;bwd", "MUSR62260;fwd"]
        expected_name = "MUSR62260;bwd-MUSR62260;fwd"

        equiv_name = self.model.create_equivalent_workspace_name(workspace_list)

        self.assertEqual(equiv_name, expected_name)

    def test_get_ws_fit_function_returns_correct_function_for_single_fit(self):
        workspace = ["MUSR62260;fwd"]
        trial_function_1 = FunctionFactory.createInitialized('name = Quadratic, A0 = 0, A1 = 0, A2 = 0')
        trial_function_2 = trial_function_1.clone()
        self.model.ws_fit_function_map = {"MUSR62260;bwd": trial_function_1, "MUSR62260;fwd": trial_function_2}

        return_function = self.model.get_ws_fit_function(workspace)

        self.assertEqual(return_function, trial_function_2)

    def test_get_ws_fit_function_returns_correct_function_if_simultaneous_fit(self):
        workspaces = ["MUSR62260;fwd", "MUSR62260;bwd"]
        trial_function_1 = FunctionFactory.createInitialized('name = Quadratic, A0 = 0, A1 = 0, A2 = 0')
        trial_function_2 = trial_function_1.clone()
        self.model.ws_fit_function_map = {"MUSR62260;fwd-MUSR62260;bwd": trial_function_1,
                                          "MUSR62261;fwd-MUSR62261;bwd": trial_function_2}

        return_function = self.model.get_ws_fit_function(workspaces)

        self.assertEqual(return_function, trial_function_1)


if __name__ == '__main__':
    unittest.main(buffer=False, verbosity=2)
