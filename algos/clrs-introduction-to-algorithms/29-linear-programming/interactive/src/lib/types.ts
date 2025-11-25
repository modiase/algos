export type Operator = "<=" | ">=" | "=";
export type Objective = "maximize" | "minimize";

export interface Constraint {
    id: string;
    coefficients: number[];
    operator: Operator;
    rhs: number;
}

export interface LPProblem {
    objective: Objective;
    objectiveCoefficients: number[];
    constraints: Constraint[];
    variableNames: string[];
    unrestrictedVariables: boolean[];
}

export interface StandardForm {
    objective: Objective;
    objectiveCoefficients: number[];
    constraints: Constraint[];
    variableNames: string[];
}

export interface SlackForm {
    objective: Objective;
    objectiveCoefficients: number[];
    constraints: Constraint[];
    variableNames: string[];
    slackVariables: string[];
}

export type FormType = "given" | "standard" | "slack";
export type DisplayMode = "compact" | "full";
