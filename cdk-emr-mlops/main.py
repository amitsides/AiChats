import * as cdk from 'aws-cdk-lib';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as emr from 'aws-cdk-lib/aws-emr';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as sagemaker from 'aws-cdk-lib/aws-sagemaker';
import { Construct } from 'constructs';

export class EmrSagemakerCvStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    // Create VPC
    const vpc = new ec2.Vpc(this, 'EmrVPC', {
      maxAzs: 2,
      natGateways: 1
    });

    // Create S3 buckets
    const dataBucket = new s3.Bucket(this, 'DataBucket', {
      versioned: true,
      encryption: s3.BucketEncryption.S3_MANAGED,
      removalPolicy: cdk.RemovalPolicy.RETAIN
    });

    const scriptsBucket = new s3.Bucket(this, 'ScriptsBucket', {
      removalPolicy: cdk.RemovalPolicy.DESTROY
    });

    // EMR Service Role
    const emrServiceRole = new iam.Role(this, 'EMRServiceRole', {
      assumedBy: new iam.ServicePrincipal('elasticmapreduce.amazonaws.com'),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName('service-role/AmazonElasticMapReduceRole')
      ]
    });

    // EMR Job Flow Role
    const emrJobFlowRole = new iam.Role(this, 'EMRJobFlowRole', {
      assumedBy: new iam.ServicePrincipal('ec2.amazonaws.com'),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName('service-role/AmazonElasticMapReduceforEC2Role')
      ]
    });

    // EMR Instance Profile
    const emrInstanceProfile = new iam.CfnInstanceProfile(this, 'EMRInstanceProfile', {
      roles: [emrJobFlowRole.roleName]
    });

    // SageMaker Execution Role
    const sagemakerRole = new iam.Role(this, 'SageMakerExecutionRole', {
      assumedBy: new iam.ServicePrincipal('sagemaker.amazonaws.com'),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonSageMakerFullAccess')
      ]
    });

    // Grant S3 access
    dataBucket.grantReadWrite(emrJobFlowRole);
    dataBucket.grantReadWrite(sagemakerRole);
    scriptsBucket.grantRead(emrJobFlowRole);

    // Security Group for EMR Cluster
    const emrSG = new ec2.SecurityGroup(this, 'EmrSecurityGroup', {
      vpc,
      description: 'Security group for EMR cluster',
      allowAllOutbound: true
    });

    // EMR Cluster Configuration
    const cluster = new emr.CfnCluster(this, 'EMRCluster', {
      name: 'DeltaLakeCluster',
      releaseLabel: 'emr-6.9.0',
      serviceRole: emrServiceRole.roleName,
      jobFlowRole: emrInstanceProfile.attrArn,
      applications: [
        { name: 'Spark' },
        { name: 'Hadoop' },
        { name: 'Hive' }
      ],
      configurations: [
        {
          classification: 'spark-defaults',
          configurationProperties: {
            'spark.jars.packages': 'io.delta:delta-core_2.12:2.1.0',
            'spark.sql.extensions': 'io.delta.sql.DeltaSparkSessionExtension',
            'spark.sql.catalog.spark_catalog': 'org.apache.spark.sql.delta.catalog.DeltaCatalog'
          }
        }
      ],
      instances: {
        masterInstanceGroup: {
          instanceCount: 1,
          instanceType: 'm5.xlarge',
          market: 'ON_DEMAND',
          name: 'Master'
        },
        coreInstanceGroup: {
          instanceCount: 2,
          instanceType: 'm5.xlarge',
          market: 'ON_DEMAND',
          name: 'Core'
        },
        ec2SubnetId: vpc.privateSubnets[0].subnetId,
        emrManagedMasterSecurityGroup: emrSG.securityGroupId,
        emrManagedSlaveSecurityGroup: emrSG.securityGroupId
      },
      tags: [
        {
          key: 'Environment',
          value: 'Development'
        }
      ]
    });

    // SageMaker Notebook Instance
    new sagemaker.CfnNotebookInstance(this, 'CVNotebookInstance', {
      instanceType: 'ml.t3.xlarge',
      roleArn: sagemakerRole.roleArn,
      subnetId: vpc.privateSubnets[0].subnetId,
      securityGroupIds: [emrSG.securityGroupId],
      notebookInstanceName: 'cv-processing-notebook',
      directInternetAccess: 'Disabled',
      volumeSizeInGb: 50
    });

    // Output the bucket names
    new cdk.CfnOutput(this, 'DataBucketName', {
      value: dataBucket.bucketName,
      description: 'Name of the data bucket'
    });

    new cdk.CfnOutput(this, 'ScriptsBucketName', {
      value: scriptsBucket.bucketName,
      description: 'Name of the scripts bucket'
    });
  }
}