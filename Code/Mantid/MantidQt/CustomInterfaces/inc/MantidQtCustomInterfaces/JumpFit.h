#ifndef MANTIDQTCUSTOMINTERFACES_JUMPFIT_H_
#define MANTIDQTCUSTOMINTERFACES_JUMPFIT_H_

#include "ui_JumpFit.h"
#include "MantidQtCustomInterfaces/IndirectBayesTab.h"

namespace MantidQt
{
	namespace CustomInterfaces
	{
		class DLLExport JumpFit : public IndirectBayesTab
		{
			Q_OBJECT

		public:
			JumpFit(QWidget * parent = 0);

		private slots:
			/// Handle when the sample input is ready
			void handleSampleInputReady(const QString& filename);
			/// Slot for when the min range on the range selector changes
			virtual void minValueChanged(double min);
			/// Slot for when the min range on the range selector changes
			virtual void maxValueChanged(double max);
			/// Slot to update the guides when the range properties change
			void updateProperties(QtProperty* prop, double val);

		private:
			/// Inherited methods from IndirectBayesTab
			virtual QString help() { return "JumpFit"; };
			virtual bool validate();
			virtual void run();

			//The ui form
			Ui::JumpFit m_uiForm;

		};
	} // namespace CustomInterfaces
} // namespace MantidQt

#endif
