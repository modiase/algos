import type { LPProblem, StandardForm, SlackForm, Constraint } from "./types";

export function toStandardForm(problem: LPProblem): StandardForm {
    const isMinimize = problem.objective === "minimize";
    const numOriginalVars = problem.variableNames.length;

    const baseObjectiveCoefficients = isMinimize
        ? problem.objectiveCoefficients.map((c) => -c)
        : problem.objectiveCoefficients;

    const expandedObjectiveCoefficients: number[] = [];
    const expandedVariableNames: string[] = [];

    for (let i = 0; i < numOriginalVars; i++) {
        if (problem.unrestrictedVariables[i]) {
            expandedObjectiveCoefficients.push(baseObjectiveCoefficients[i]);
            expandedObjectiveCoefficients.push(-baseObjectiveCoefficients[i]);
            expandedVariableNames.push(`x_{${i + 1}}^+`);
            expandedVariableNames.push(`x_{${i + 1}}^-`);
        } else {
            expandedObjectiveCoefficients.push(baseObjectiveCoefficients[i]);
            expandedVariableNames.push(problem.variableNames[i]);
        }
    }

    const expandConstraintCoefficients = (coeffs: number[]): number[] => {
        const expanded: number[] = [];
        for (let i = 0; i < numOriginalVars; i++) {
            if (problem.unrestrictedVariables[i]) {
                expanded.push(coeffs[i]);
                expanded.push(-coeffs[i]);
            } else {
                expanded.push(coeffs[i]);
            }
        }
        return expanded;
    };

    const constraints: Constraint[] = [];

    problem.constraints.forEach((c) => {
        const expandedCoeffs = expandConstraintCoefficients(c.coefficients);

        if (c.operator === ">=") {
            constraints.push({
                ...c,
                coefficients: expandedCoeffs.map((v) => -v),
                operator: "<=",
                rhs: -c.rhs,
            });
        } else if (c.operator === "=") {
            constraints.push({
                ...c,
                coefficients: expandedCoeffs,
                operator: "<=",
                rhs: c.rhs,
            });
            constraints.push({
                ...c,
                coefficients: expandedCoeffs.map((v) => -v),
                operator: "<=",
                rhs: -c.rhs,
            });
        } else {
            constraints.push({
                ...c,
                coefficients: expandedCoeffs,
                operator: "<=",
                rhs: c.rhs,
            });
        }
    });

    return {
        objective: "maximize",
        objectiveCoefficients: expandedObjectiveCoefficients,
        constraints,
        variableNames: expandedVariableNames,
    };
}

export function toSlackForm(standard: StandardForm): SlackForm {
    const slackVariables: string[] = [];
    const newConstraints: Constraint[] = [];
    const numOriginalVars = standard.variableNames.length;

    standard.constraints.forEach((c, i) => {
        const slackName = `s_{${i + 1}}`;
        slackVariables.push(slackName);

        const newCoefficients = [...c.coefficients];

        for (let j = 0; j < slackVariables.length - 1; j++) {
            newCoefficients.push(0);
        }
        newCoefficients.push(1);

        newConstraints.push({
            ...c,
            coefficients: newCoefficients,
            operator: "=",
        });
    });

    const extendedObjective = [
        ...standard.objectiveCoefficients,
        ...new Array(slackVariables.length).fill(0),
    ];

    return {
        objective: standard.objective,
        objectiveCoefficients: extendedObjective,
        constraints: newConstraints,
        variableNames: [...standard.variableNames, ...slackVariables],
        slackVariables,
    };
}
