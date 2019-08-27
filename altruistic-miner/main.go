package main

import (
	"context"

	"github.com/pegnet/pegnet/balances"
	pegnet "github.com/pegnet/pegnet/cmd"
	"github.com/spf13/cobra"
)

func init() {
	pegnet.RootCmd.AddCommand()
}

func main() {
	pegnet.Execute()
}

// The cli enter point
var altruisticMiner = &cobra.Command{
	Use:   "pegnet",
	Short: "pegnet is the cli tool to run or interact with a PegNet node",
	Run: func(cmd *cobra.Command, args []string) {
		ctx, cancel := context.WithCancel(context.Background())
		b := balances.NewBalanceTracker()

		pegnet.ValidateConfig(pegnet.Config) // Will fatal log if it fails

		// Services
		monitor := pegnet.LaunchFactomMonitor(pegnet.Config)
		grader := pegnet.LaunchGrader(pegnet.Config, monitor, b, ctx, true)
		statTracker := pegnet.LaunchStatistics(pegnet.Config, ctx)
		apiserver := pegnet.LaunchAPI(pegnet.Config, statTracker, grader, b, true)
		pegnet.LaunchControlPanel(pegnet.Config, ctx, monitor, statTracker, b)
		var _ = apiserver

		// This is a blocking call
		coord := pegnet.LaunchMiners(pegnet.Config, ctx, monitor, grader, statTracker)

		// Calling cancel() will cancel the stat tracker collection AND the miners
		var _, _ = cancel, coord
	},
}
